"""Headless multi-project & multi-agent discovery, verbal short names, and state persistence."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_AGENT_ID = "101"

IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".tox",
        ".eggs",
    }
)

_TIER_EXACT = 1
_TIER_NORM_EXACT = 2
_TIER_PREFIX = 3
_TIER_SUBSTRING = 4
_TIER_TOKEN_SUBSET = 5
_TIER_NO_MATCH = 99

_KIND_WHOLE = 0
_KIND_STRIPPED_PREFIX = 1
_KIND_COMBO_COMMA = 2
_KIND_COMBO_COMMA_STRIPPED = 3
_KIND_COMBO_SPACE = 4
_KIND_COMBO_SPACE_STRIPPED = 5

_RE_LEADING_PROJECT = re.compile(r"^\s*project\s+(.+?)\s*$", re.IGNORECASE)
_RE_COMBO_COMMA = re.compile(
    r"^\s*(?P<proj>.+?)\s*,\s*(?:agent(?:\s+#?|\s*#\s*)|#\s*)(?P<agent>[a-zA-Z0-9_-]+)\s*$",
    re.IGNORECASE,
)
_RE_COMBO_SPACE = re.compile(
    r"^\s*(?P<proj>.+?)\s+(?:agent(?:\s+#?|\s*#\s*)|#\s*)(?P<agent>[a-zA-Z0-9_-]+)\s*$",
    re.IGNORECASE,
)
_RE_ONLY_AGENT = re.compile(
    r"^\s*(?:agent(?:\s+#?|\s*#\s*)|#\s*)(?P<agent>[a-zA-Z0-9_-]+)\s*$",
    re.IGNORECASE,
)


def slugify_project_id(name: str) -> str:
    """Convert a directory or project name into a stable URL/ID slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "workspace"


def generate_short_name(dir_name: str) -> str:
    """Generate a clean verbal short name from a project directory name."""
    raw = dir_name.strip()
    if not raw:
        return "Workspace"

    tokens = [tok for tok in re.split(r"[-_\s]+", raw) if tok]
    words: list[str] = []
    for tok in tokens:
        split_camel = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tok)
        split_camel = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", split_camel)
        words.extend(w for w in split_camel.split() if w)

    if len(words) >= 3 and words[0].lower() == "antigravity":
        words = words[1:]

    if not words:
        return "Workspace"

    formatted: list[str] = []
    for word in words:
        if word.isupper() and len(word) >= 2:
            formatted.append(word)
        else:
            formatted.append(word.capitalize())
    return " ".join(formatted)


def format_project_agent(short_name: str, agent_id: str) -> str:
    """Format canonical verbal project-agent identifier."""
    return f"project {short_name}, agent {agent_id}"


def _is_git_repo(path: Path) -> bool:
    """Return True if `path` is a real git repository or git worktree root."""
    try:
        dot_git = path / ".git"
        if dot_git.is_dir():
            return (dot_git / "HEAD").is_file()
        if dot_git.is_file():
            raw = dot_git.read_text(encoding="utf-8", errors="ignore")
            first_line = raw.splitlines()[0].strip() if raw else ""
            if not first_line.lower().startswith("gitdir:"):
                return False
            gitdir_raw = first_line.split(":", 1)[1].strip()
            if not gitdir_raw:
                return False
            target = Path(gitdir_raw)
            if not target.is_absolute():
                target = path / target
            target = target.resolve()
            return target.is_dir() and (target / "HEAD").is_file()
    except (OSError, ValueError):
        return False
    return False


def inspect_git_metadata(project_path: Path, timeout: float = 2.0) -> tuple[str | None, int]:
    """Safely inspect git branch and uncommitted change count for a project directory."""
    if not _is_git_repo(project_path):
        return (None, 0)

    try:
        branch_res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        branch: str | None = None
        if branch_res.returncode == 0:
            raw_branch = branch_res.stdout.strip()
            if raw_branch == "HEAD":
                sha_res = subprocess.run(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=str(project_path),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
                sha = sha_res.stdout.strip() if sha_res.returncode == 0 else ""
                branch = f"detached@{sha}" if sha else "HEAD (detached)"
            elif raw_branch:
                branch = raw_branch
        else:
            sym_res = subprocess.run(
                ["git", "symbolic-ref", "--short", "HEAD"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if sym_res.returncode == 0 and sym_res.stdout.strip():
                branch = sym_res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(project_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        uncommitted = (
            len([line for line in status_res.stdout.splitlines() if line.strip()])
            if status_res.returncode == 0
            else 0
        )
        return (branch, uncommitted)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return (None, 0)


@dataclass
class ProjectAgentState:
    """Runtime and persisted state for one agent slot inside a project."""

    agent_id: str
    short_name: str
    project_agent: str
    conversation_id: str | None = None
    status: str = "idle"
    current_task: str | None = None
    last_summary: str | None = None
    swarm_nodes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["swarm_nodes"] = [dict(n) for n in self.swarm_nodes]
        return data


@dataclass
class ProjectRecord:
    """Discovered workspace project metadata and agent registry."""

    id: str
    name: str
    short_name: str
    path: str
    git_branch: str | None = None
    uncommitted_changes: int = 0
    active_agent_id: str = DEFAULT_AGENT_ID
    agents: dict[str, ProjectAgentState] = field(default_factory=dict)


class ProjectManager:
    """Scans workspace projects, tracks concurrent project agents, and persists thread state."""

    def __init__(
        self,
        workspace_dir: str | Path,
        save_dir: str | Path,
        state_file: str | Path | None = None,
        default_agent_id: str = DEFAULT_AGENT_ID,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.state_file = (
            Path(state_file).expanduser().resolve()
            if state_file
            else (self.save_dir / "projects_state.json")
        )
        self.default_agent_id = str(default_agent_id).strip() or DEFAULT_AGENT_ID
        self._lock = threading.RLock()
        self._projects: dict[str, ProjectRecord] = {}
        self._persisted_projects: dict[str, dict[str, Any]] = {}
        self._active_task_tokens: dict[tuple[str, str], str] = {}
        self._active_project_id: str | None = None
        self._active_agent_id: str = self.default_agent_id

        self._load_state()
        self.scan_projects()

    @staticmethod
    def _normalize_swarm_nodes(raw_nodes: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_nodes, list):
            return []
        return [dict(node) for node in raw_nodes if isinstance(node, dict)]

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            active_proj = raw.get("active_project_id")
            if isinstance(active_proj, str) and active_proj.strip():
                self._active_project_id = active_proj.strip()
            active_agent = raw.get("active_agent_id")
            if isinstance(active_agent, str) and active_agent.strip():
                self._active_agent_id = active_agent.strip()
            persisted = raw.get("projects")
            if isinstance(persisted, dict):
                loaded_projects: dict[str, dict[str, Any]] = {}
                for k, v in persisted.items():
                    if not isinstance(v, dict):
                        continue
                    proj_entry = dict(v)
                    raw_agents = proj_entry.get("agents")
                    if isinstance(raw_agents, dict):
                        norm_agents: dict[str, dict[str, Any]] = {}
                        for aid, agent_val in raw_agents.items():
                            if not isinstance(agent_val, dict):
                                continue
                            agent_copy = dict(agent_val)
                            agent_copy["swarm_nodes"] = self._normalize_swarm_nodes(
                                agent_copy.get("swarm_nodes")
                            )
                            norm_agents[str(aid)] = agent_copy
                        proj_entry["agents"] = norm_agents
                    loaded_projects[str(k)] = proj_entry
                self._persisted_projects = loaded_projects
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "Could not load project state from %s: %s", self.state_file, exc
            )

    def _sync_persisted_snapshot_locked(self) -> dict[str, Any]:
        merged_projects: dict[str, dict[str, Any]] = {
            pid: dict(pdata) for pid, pdata in self._persisted_projects.items()
        }
        for pid, rec in self._projects.items():
            prev = dict(merged_projects.get(pid) or {})
            prev_agents = dict(prev.get("agents") or {})
            for aid, agent in rec.agents.items():
                prev_agents[aid] = {
                    "conversation_id": agent.conversation_id,
                    "last_summary": agent.last_summary,
                    "swarm_nodes": [dict(n) for n in agent.swarm_nodes],
                }
            merged_projects[pid] = {
                "short_name": rec.short_name,
                "active_agent_id": rec.active_agent_id,
                "agents": prev_agents,
            }
        self._persisted_projects = merged_projects
        return {
            "active_project_id": self._active_project_id,
            "active_agent_id": self._active_agent_id,
            "projects": merged_projects,
        }

    def _save_state(self) -> None:
        with self._lock:
            payload = self._sync_persisted_snapshot_locked()
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.state_file.parent,
                    prefix=".projects_state_",
                    suffix=".tmp",
                    delete=False,
                ) as tmp_file:
                    tmp_path = tmp_file.name
                    json.dump(payload, tmp_file, indent=2)
                    tmp_file.flush()
                    os.fsync(tmp_file.fileno())
                os.replace(tmp_path, self.state_file)
            except Exception as exc:
                logger.warning("Failed to persist project state: %s", exc)
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    def _ensure_agent_slot(
        self, record: ProjectRecord, agent_id: str
    ) -> ProjectAgentState:
        aid = str(agent_id).strip() or self.default_agent_id
        agent = record.agents.get(aid)
        if agent is None:
            persisted_agents = (
                self._persisted_projects.get(record.id, {}).get("agents") or {}
            )
            persisted_agent = (
                persisted_agents.get(aid, {})
                if isinstance(persisted_agents, dict)
                else {}
            )
            agent = ProjectAgentState(
                agent_id=aid,
                short_name=f"agent {aid}",
                project_agent=format_project_agent(record.short_name, aid),
                conversation_id=persisted_agent.get("conversation_id"),
                last_summary=persisted_agent.get("last_summary"),
                swarm_nodes=self._normalize_swarm_nodes(
                    persisted_agent.get("swarm_nodes")
                ),
            )
            record.agents[aid] = agent
        else:
            agent.short_name = f"agent {aid}"
            agent.project_agent = format_project_agent(record.short_name, aid)
        return agent

    def _serialize_project(self, record: ProjectRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "name": record.name,
            "short_name": record.short_name,
            "path": record.path,
            "git_branch": record.git_branch,
            "uncommitted_changes": record.uncommitted_changes,
            "active": record.id == self._active_project_id,
            "active_agent_id": record.active_agent_id,
            "agents": [
                agent.to_dict()
                for agent in sorted(record.agents.values(), key=lambda a: a.agent_id)
            ],
        }

    def scan_projects(self) -> list[dict[str, Any]]:
        """Scan `workspace_dir` for projects, skipping ignored/state dirs with fallback."""
        with self._lock:
            if not self.workspace_dir.exists() or not self.workspace_dir.is_dir():
                self._projects.clear()
                self._active_project_id = None
                return []

            if _is_git_repo(self.workspace_dir):
                candidate_dirs = [self.workspace_dir]
            else:
                candidate_dirs = []
                for entry in sorted(
                    self.workspace_dir.iterdir(), key=lambda p: p.name.lower()
                ):
                    if not entry.is_dir():
                        continue
                    if entry.name.startswith(".") or entry.name in IGNORED_DIR_NAMES:
                        continue
                    resolved = entry.resolve()
                    is_state_dir = (
                        resolved == self.save_dir or resolved == self.state_file.parent
                    )
                    is_state_ancestor = (
                        resolved != self.workspace_dir
                        and not _is_git_repo(resolved)
                        and (
                            self.save_dir.is_relative_to(resolved)
                            or self.state_file.parent.is_relative_to(resolved)
                        )
                    )
                    if is_state_dir or is_state_ancestor:
                        continue
                    candidate_dirs.append(entry)

                if not candidate_dirs:
                    candidate_dirs = [self.workspace_dir]

            new_projects: dict[str, ProjectRecord] = {}
            used_ids: set[str] = set()
            used_short_names: set[str] = set()

            for proj_dir in candidate_dirs:
                base_id = slugify_project_id(proj_dir.name)
                proj_id = base_id
                suffix = 2
                while proj_id in used_ids:
                    proj_id = f"{base_id}-{suffix}"
                    suffix += 1
                used_ids.add(proj_id)

                persisted_proj = self._persisted_projects.get(proj_id, {})
                existing_rec = self._projects.get(proj_id)

                raw_short = (
                    existing_rec.short_name
                    if existing_rec is not None
                    else persisted_proj.get("short_name")
                )
                base_short = (
                    str(raw_short).strip()
                    if isinstance(raw_short, str) and raw_short.strip()
                    else generate_short_name(proj_dir.name)
                )
                short_name = base_short
                sn_suffix = 2
                while short_name.lower() in used_short_names:
                    short_name = f"{base_short} {sn_suffix}"
                    sn_suffix += 1
                used_short_names.add(short_name.lower())

                git_branch, uncommitted = inspect_git_metadata(proj_dir)

                agents: dict[str, ProjectAgentState] = {}
                if existing_rec is not None:
                    for aid, st in existing_rec.agents.items():
                        agents[aid] = ProjectAgentState(
                            agent_id=aid,
                            short_name=f"agent {aid}",
                            project_agent=format_project_agent(short_name, aid),
                            conversation_id=st.conversation_id,
                            status=st.status,
                            current_task=st.current_task,
                            last_summary=st.last_summary,
                            swarm_nodes=[dict(n) for n in st.swarm_nodes],
                        )
                else:
                    persisted_agents = persisted_proj.get("agents") or {}
                    if isinstance(persisted_agents, dict):
                        for aid, p_agent in persisted_agents.items():
                            if not isinstance(p_agent, dict):
                                continue
                            aid_str = str(aid).strip()
                            if not aid_str:
                                continue
                            agents[aid_str] = ProjectAgentState(
                                agent_id=aid_str,
                                short_name=f"agent {aid_str}",
                                project_agent=format_project_agent(short_name, aid_str),
                                conversation_id=p_agent.get("conversation_id"),
                                last_summary=p_agent.get("last_summary"),
                                swarm_nodes=self._normalize_swarm_nodes(
                                    p_agent.get("swarm_nodes")
                                ),
                            )

                active_aid = (
                    existing_rec.active_agent_id
                    if existing_rec is not None
                    else str(
                        persisted_proj.get("active_agent_id") or self.default_agent_id
                    ).strip()
                    or self.default_agent_id
                )

                rec = ProjectRecord(
                    id=proj_id,
                    name=proj_dir.name,
                    short_name=short_name,
                    path=str(proj_dir.resolve()),
                    git_branch=git_branch,
                    uncommitted_changes=uncommitted,
                    active_agent_id=active_aid,
                    agents=agents,
                )
                self._ensure_agent_slot(rec, self.default_agent_id)
                self._ensure_agent_slot(rec, rec.active_agent_id)
                new_projects[proj_id] = rec

            self._projects = new_projects
            self._active_task_tokens = {
                key: tok
                for key, tok in self._active_task_tokens.items()
                if key[0] in self._projects
            }

            prev_active_project_id = self._active_project_id
            if self._active_project_id not in self._projects:
                self._active_project_id = (
                    next(iter(self._projects.keys())) if self._projects else None
                )
            active_project_switched = self._active_project_id != prev_active_project_id

            if self._active_project_id is not None:
                active_rec = self._projects[self._active_project_id]
                if active_project_switched or self._active_agent_id not in active_rec.agents:
                    self._active_agent_id = active_rec.active_agent_id
                else:
                    active_rec.active_agent_id = self._active_agent_id

            return [self._serialize_project(rec) for rec in self._projects.values()]

    @staticmethod
    def _clean_agent_token(raw_agent: str | None) -> str | None:
        if raw_agent is None:
            return None
        cleaned = re.sub(
            r"^\s*(?:agent(?:\s+#?|\s*#\s*)|#\s*)",
            "",
            str(raw_agent).strip(),
            flags=re.IGNORECASE,
        )
        return cleaned.strip() or None

    @staticmethod
    def _find_project_matches(
        records: list[ProjectRecord], proj_query: str
    ) -> tuple[int, list[ProjectRecord]]:
        q = proj_query.strip().lower()
        if not q:
            return (_TIER_NO_MATCH, [])
        q_norm = re.sub(r"[^a-z0-9]+", " ", q).strip()

        # Tier 1: Exact case-insensitive match
        tier1 = [
            rec
            for rec in records
            if q in (rec.id.lower(), rec.name.lower(), rec.short_name.lower())
        ]
        if tier1:
            return (_TIER_EXACT, tier1)

        # Tier 2: Normalized exact match
        if q_norm:
            tier2: list[ProjectRecord] = []
            for rec in records:
                norms = {
                    re.sub(r"[^a-z0-9]+", " ", cand.lower()).strip()
                    for cand in (rec.id, rec.name, rec.short_name)
                }
                if q_norm in norms:
                    tier2.append(rec)
            if tier2:
                return (_TIER_NORM_EXACT, tier2)

        # Tier 3 (Prefix) & Tier 4 (Substring / reverse whole-token containment)
        prefix_matches: list[ProjectRecord] = []
        substring_matches: list[ProjectRecord] = []
        q_token_set = set(q_norm.split()) if q_norm else set()
        for rec in records:
            fields = [
                rec.short_name.lower(),
                rec.name.lower(),
                rec.id.lower(),
                re.sub(r"[^a-z0-9]+", " ", rec.short_name.lower()).strip(),
                re.sub(r"[^a-z0-9]+", " ", rec.name.lower()).strip(),
            ]
            matched_prefix = False
            matched_substring = False
            for val in fields:
                if not val:
                    continue
                if val.startswith(q) or (q_norm and val.startswith(q_norm)):
                    matched_prefix = True
                    break
                if q in val or (q_norm and q_norm in val):
                    matched_substring = True
                elif q_token_set:
                    val_norm = re.sub(r"[^a-z0-9]+", " ", val).strip()
                    val_tokens = val_norm.split()
                    if val_tokens and all(tok in q_token_set for tok in val_tokens):
                        matched_substring = True
            if matched_prefix:
                prefix_matches.append(rec)
            elif matched_substring:
                substring_matches.append(rec)

        if prefix_matches:
            return (_TIER_PREFIX, prefix_matches)
        if substring_matches:
            return (_TIER_SUBSTRING, substring_matches)

        # Tier 5: Token subset match
        q_tokens = q_norm.split()
        if q_tokens:
            token_matches: list[ProjectRecord] = []
            for rec in records:
                hay = re.sub(
                    r"[^a-z0-9]+",
                    " ",
                    f"{rec.id} {rec.name} {rec.short_name}".lower(),
                )
                hay_tokens = set(hay.split())
                if all(tok in hay_tokens for tok in q_tokens):
                    token_matches.append(rec)
            if token_matches:
                return (_TIER_TOKEN_SUBSET, token_matches)

        return (_TIER_NO_MATCH, [])

    @classmethod
    def _resolve_phrase_against_records(
        cls,
        records: list[ProjectRecord],
        phrase: str,
        *,
        allow_pure_agent: bool,
    ) -> tuple[ProjectRecord | None, str | None]:
        text = phrase.strip()
        if not text:
            return (None, None)

        m_lead = _RE_LEADING_PROJECT.match(text)
        stripped_text = m_lead.group(1).strip() if m_lead else None
        m_comma = _RE_COMBO_COMMA.match(text)

        pure_agent_match = _RE_ONLY_AGENT.match(text)
        if pure_agent_match is None and stripped_text:
            pure_agent_match = _RE_ONLY_AGENT.match(stripped_text)
        if pure_agent_match is None and m_comma:
            if m_comma.group("proj").strip().lower() == "project":
                pure_agent_match = m_comma
        is_pure_agent_phrase = allow_pure_agent and (pure_agent_match is not None)

        hypotheses: list[tuple[str, str | None, int, int]] = []
        if m_comma:
            hypotheses.append((text, None, _KIND_WHOLE, _TIER_EXACT))
            proj_part = m_comma.group("proj").strip()
            agent_part = m_comma.group("agent").strip()
            comma_max = (
                _TIER_NORM_EXACT
                if proj_part.lower() == "project"
                else _TIER_TOKEN_SUBSET
            )
            hypotheses.append((proj_part, agent_part, _KIND_COMBO_COMMA, comma_max))
            m_comma_lead = _RE_LEADING_PROJECT.match(proj_part)
            if m_comma_lead:
                stripped_proj = m_comma_lead.group(1).strip()
                stripped_max = (
                    _TIER_NORM_EXACT
                    if stripped_proj.lower() == "project"
                    else _TIER_TOKEN_SUBSET
                )
                hypotheses.append(
                    (
                        stripped_proj,
                        agent_part,
                        _KIND_COMBO_COMMA_STRIPPED,
                        stripped_max,
                    )
                )
        else:
            whole_max = _TIER_PREFIX if is_pure_agent_phrase else _TIER_TOKEN_SUBSET
            hypotheses.append((text, None, _KIND_WHOLE, whole_max))
            if stripped_text:
                hypotheses.append(
                    (stripped_text, None, _KIND_STRIPPED_PREFIX, whole_max)
                )
            if not is_pure_agent_phrase:
                m_space = _RE_COMBO_SPACE.match(text)
                if m_space:
                    proj_part = m_space.group("proj").strip()
                    agent_part = m_space.group("agent").strip()
                    if proj_part and proj_part.lower() != "project":
                        hypotheses.append(
                            (
                                proj_part,
                                agent_part,
                                _KIND_COMBO_SPACE,
                                _TIER_TOKEN_SUBSET,
                            )
                        )
                        m_space_lead = _RE_LEADING_PROJECT.match(proj_part)
                        if m_space_lead:
                            stripped_proj = m_space_lead.group(1).strip()
                            if stripped_proj and stripped_proj.lower() != "project":
                                hypotheses.append(
                                    (
                                        stripped_proj,
                                        agent_part,
                                        _KIND_COMBO_SPACE_STRIPPED,
                                        _TIER_TOKEN_SUBSET,
                                    )
                                )

        best_score: tuple[int, int] | None = None
        best_result: tuple[str, list[ProjectRecord], str | None] | None = None

        for proj_cand, agent_tok, kind_prio, max_tier in hypotheses:
            if not proj_cand:
                continue
            tier, matches = cls._find_project_matches(records, proj_cand)
            if not matches or tier > max_tier:
                continue
            score = (tier, kind_prio)
            if best_score is None or score < best_score:
                best_score = score
                best_result = (proj_cand, matches, agent_tok)

        if best_result is not None:
            proj_cand, matches, agent_tok = best_result
            if len(matches) > 1:
                names = ", ".join(f"{r.short_name} ({r.id})" for r in matches)
                raise ValueError(
                    f"Ambiguous project query {proj_cand!r}; matches: {names}"
                )
            return (matches[0], agent_tok)

        if is_pure_agent_phrase and pure_agent_match is not None:
            return (None, pure_agent_match.group("agent").strip())

        raise ValueError(f"Unknown project: {text!r}")

    @classmethod
    def _resolve_target_from_records(
        cls,
        records: list[ProjectRecord],
        *,
        project: str | None = None,
        agent: str | None = None,
        project_agent: str | None = None,
    ) -> tuple[ProjectRecord | None, str | None]:
        raw_proj = (
            project.strip() if isinstance(project, str) and project.strip() else None
        )
        raw_pa = (
            project_agent.strip()
            if isinstance(project_agent, str) and project_agent.strip()
            else None
        )
        explicit_agent = (
            cls._clean_agent_token(agent)
            if isinstance(agent, str) and agent.strip()
            else None
        )

        pa_rec: ProjectRecord | None = None
        pa_agent: str | None = None
        if raw_pa:
            pa_rec, pa_agent = cls._resolve_phrase_against_records(
                records, raw_pa, allow_pure_agent=True
            )

        proj_rec: ProjectRecord | None = None
        proj_agent: str | None = None
        if raw_proj:
            proj_rec, proj_agent = cls._resolve_phrase_against_records(
                records, raw_proj, allow_pure_agent=(explicit_agent is None)
            )

        final_rec = proj_rec if proj_rec is not None else pa_rec
        final_agent = explicit_agent or proj_agent or pa_agent
        return (final_rec, final_agent)

    @classmethod
    def _parse_target_query(
        cls,
        project: str | None = None,
        agent: str | None = None,
        project_agent: str | None = None,
        projects: dict[str, ProjectRecord] | None = None,
    ) -> tuple[str | None, str | None]:
        if projects is not None:
            rec, parsed_agent = cls._resolve_target_from_records(
                list(projects.values()),
                project=project,
                agent=agent,
                project_agent=project_agent,
            )
            return (rec.id if rec is not None else None, parsed_agent)

        def _parse_phrase_syntactic(
            phrase: str, *, allow_pure_agent: bool
        ) -> tuple[str | None, str | None]:
            text = phrase.strip()
            if not text:
                return (None, None)

            m_lead = _RE_LEADING_PROJECT.match(text)
            stripped_text = m_lead.group(1).strip() if m_lead else None

            if allow_pure_agent:
                m_agent = _RE_ONLY_AGENT.match(text)
                if m_agent:
                    return (None, m_agent.group("agent").strip())
                if stripped_text:
                    m_agent_stripped = _RE_ONLY_AGENT.match(stripped_text)
                    if m_agent_stripped:
                        return (None, m_agent_stripped.group("agent").strip())

            m_comma = _RE_COMBO_COMMA.match(text)
            if m_comma:
                proj_part = m_comma.group("proj").strip()
                agent_part = m_comma.group("agent").strip()
                m2 = _RE_LEADING_PROJECT.match(proj_part)
                if m2:
                    proj_part = m2.group(1).strip()
                if proj_part.lower() == "project":
                    return (None, agent_part)
                return (proj_part or None, agent_part)

            m_space = _RE_COMBO_SPACE.match(text)
            if m_space:
                proj_part = m_space.group("proj").strip()
                agent_part = m_space.group("agent").strip()
                m2 = _RE_LEADING_PROJECT.match(proj_part)
                if m2:
                    proj_part = m2.group(1).strip()
                if proj_part.lower() == "project":
                    return (None, agent_part)
                return (proj_part or None, agent_part)

            if stripped_text:
                return (stripped_text, None)
            return (text, None)

        parsed_proj: str | None = None
        parsed_agent: str | None = None

        has_explicit_pa = isinstance(project_agent, str) and bool(project_agent.strip())
        explicit_agent = (
            cls._clean_agent_token(agent)
            if isinstance(agent, str) and agent.strip()
            else None
        )

        if has_explicit_pa and project_agent is not None:
            parsed_proj, parsed_agent = _parse_phrase_syntactic(
                project_agent, allow_pure_agent=True
            )

        if isinstance(project, str) and project.strip():
            p_proj, p_agent = _parse_phrase_syntactic(
                project, allow_pure_agent=(explicit_agent is None)
            )
            if p_proj:
                parsed_proj = p_proj
            if p_agent and not parsed_agent:
                parsed_agent = p_agent

        if explicit_agent:
            parsed_agent = explicit_agent

        return (parsed_proj, parsed_agent)

    def _match_project(self, proj_query: str) -> ProjectRecord:
        tier, matches = self._find_project_matches(
            list(self._projects.values()), proj_query
        )
        if not matches or tier == _TIER_NO_MATCH:
            raise ValueError(f"Unknown project: {proj_query!r}")
        if len(matches) > 1:
            names = ", ".join(f"{r.short_name} ({r.id})" for r in matches)
            raise ValueError(
                f"Ambiguous project query {proj_query!r}; matches: {names}"
            )
        return matches[0]

    def _build_target_payload(
        self, rec: ProjectRecord, agent: ProjectAgentState
    ) -> dict[str, Any]:
        return {
            "project_id": rec.id,
            "project_name": rec.name,
            "short_name": rec.short_name,
            "path": rec.path,
            "agent_id": agent.agent_id,
            "project_agent": agent.project_agent,
            "conversation_id": agent.conversation_id,
            "git_branch": rec.git_branch,
            "uncommitted_changes": rec.uncommitted_changes,
        }

    def select_project(
        self,
        project: str | None = None,
        agent: str | None = None,
        project_agent: str | None = None,
    ) -> dict[str, Any]:
        """Select and persist the active project and/or project_agent."""
        with self._lock:
            if not self._projects:
                self.scan_projects()

            rec, agent_query = self._resolve_target_from_records(
                list(self._projects.values()),
                project=project,
                agent=agent,
                project_agent=project_agent,
            )

            if rec is None:
                if (
                    self._active_project_id
                    and self._active_project_id in self._projects
                ):
                    rec = self._projects[self._active_project_id]
                elif self._projects:
                    rec = next(iter(self._projects.values()))
                else:
                    raise ValueError("No projects available in workspace")

            target_agent_id = (
                agent_query or rec.active_agent_id or self.default_agent_id
            )
            target_agent = self._ensure_agent_slot(rec, target_agent_id)

            self._active_project_id = rec.id
            self._active_agent_id = target_agent.agent_id
            rec.active_agent_id = target_agent.agent_id
            self._save_state()
            return self._build_target_payload(rec, target_agent)

    def get_target(
        self,
        project: str | None = None,
        agent: str | None = None,
        project_agent: str | None = None,
    ) -> dict[str, Any]:
        """Resolve project/agent target without mutating the active selection."""
        with self._lock:
            if not self._projects:
                self.scan_projects()

            rec, agent_query = self._resolve_target_from_records(
                list(self._projects.values()),
                project=project,
                agent=agent,
                project_agent=project_agent,
            )

            if rec is None:
                if (
                    self._active_project_id
                    and self._active_project_id in self._projects
                ):
                    rec = self._projects[self._active_project_id]
                elif self._projects:
                    rec = next(iter(self._projects.values()))
                else:
                    fallback_agent = agent_query or self.default_agent_id
                    ws_short = generate_short_name(self.workspace_dir.name)
                    return {
                        "project_id": "workspace",
                        "project_name": self.workspace_dir.name,
                        "short_name": ws_short,
                        "path": str(self.workspace_dir),
                        "agent_id": fallback_agent,
                        "project_agent": format_project_agent(ws_short, fallback_agent),
                        "conversation_id": None,
                        "git_branch": None,
                        "uncommitted_changes": 0,
                    }

            default_for_rec = (
                self._active_agent_id
                if rec.id == self._active_project_id
                else rec.active_agent_id
            )
            target_agent_id = agent_query or default_for_rec or self.default_agent_id
            created = target_agent_id not in rec.agents
            target_agent = self._ensure_agent_slot(rec, target_agent_id)
            if created:
                self._save_state()
            return self._build_target_payload(rec, target_agent)

    def set_conversation_id(
        self, project_id: str, agent_id: str, conversation_id: str
    ) -> None:
        """Persist `conversation_id` for a `(project_id, agent_id)` thread."""
        if not conversation_id:
            return
        with self._lock:
            aid = str(agent_id).strip() or self.default_agent_id
            rec = self._projects.get(project_id)
            if rec is None:
                entry = self._persisted_projects.setdefault(project_id, {})
                agents = entry.setdefault("agents", {})
                if isinstance(agents, dict):
                    agent_entry = agents.setdefault(aid, {})
                    if isinstance(agent_entry, dict):
                        agent_entry["conversation_id"] = conversation_id
                self._save_state()
                return

            agent = self._ensure_agent_slot(rec, aid)
            agent.conversation_id = conversation_id
            self._save_state()

    def update_agent_status(
        self,
        project_id: str,
        agent_id: str,
        *,
        status: str,
        current_task: str | None = None,
        last_summary: str | None = None,
        swarm_nodes: list[dict[str, Any]] | None = None,
        task_id: str | None = None,
    ) -> bool:
        """Update runtime status, task, swarm nodes, and optional persisted summary.

        When ``task_id`` is provided, enforces task-generation ownership on
        ``(project_id, agent_id)`` so an older cancelled or exiting task can
        never overwrite a newer running task's status.
        """
        with self._lock:
            rec = self._projects.get(project_id)
            if rec is None:
                return False
            agent = self._ensure_agent_slot(rec, agent_id)
            token_key = (project_id, agent.agent_id)
            if status == "running":
                if task_id is not None:
                    self._active_task_tokens[token_key] = task_id
                else:
                    self._active_task_tokens.pop(token_key, None)
            else:
                if task_id is not None:
                    active_token = self._active_task_tokens.get(token_key)
                    if active_token is not None and active_token != task_id:
                        return False
                else:
                    self._active_task_tokens.pop(token_key, None)
            agent.status = status
            if current_task is not None or status in ("idle", "error", "completed"):
                agent.current_task = current_task
            should_save = False
            if swarm_nodes is not None:
                agent.swarm_nodes = self._normalize_swarm_nodes(swarm_nodes)
                should_save = True
            if last_summary is not None:
                agent.last_summary = last_summary
                should_save = True
            if should_save:
                self._save_state()
            return True

    def get_state_summary(self, *, rescan: bool = False) -> dict[str, Any]:
        """Return Contract 2 project/agent state summary."""
        with self._lock:
            if rescan:
                self.scan_projects()
            active_rec = (
                self._projects.get(self._active_project_id)
                if self._active_project_id
                else None
            )
            active_project_agent = (
                format_project_agent(active_rec.short_name, self._active_agent_id)
                if active_rec is not None
                else None
            )
            return {
                "active_project_id": self._active_project_id,
                "active_agent_id": self._active_agent_id,
                "active_project_agent": active_project_agent,
                "projects": [
                    self._serialize_project(rec) for rec in self._projects.values()
                ],
            }
