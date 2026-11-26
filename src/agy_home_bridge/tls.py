"""Automatic LAN Self-Signed Root CA & Server TLS Certificate Generator.

Mobile browsers (iOS Safari and Android Chrome) enforce a W3C Secure Context requirement:
`navigator.mediaDevices.getUserMedia` is `undefined` on plain `http://<LAN_IP>` URLs and
only exists on `https://` or `localhost`.

Meanwhile, local Google Home / Chromecast speakers fetch TTS audio over plain `http://<LAN_IP>:8765`.
By generating a local Root CA + SAN server certificate (for `192.168.1.249`, `agy-home-bridge.local`,
etc.) and serving BOTH HTTP (`:8765` for Google Home speakers + one-tap CA download) and
HTTPS (`:8443` for mobile/desktop microphone access), mobile devices get full WebAudio + WebRTC
microphone access on the LAN without needing a public domain.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

logger = logging.getLogger(__name__)


def ensure_lan_tls_certificates(
    cert_dir: Path,
    lan_ip: str,
    hostname: str = "agy-home-bridge.local",
) -> tuple[Path, Path, Path]:
    """Ensure Root CA (`ca.pem`) and LAN Server Certificate (`server.crt`, `server.key`) exist.

    Returns:
        `(ca_cert_path, server_cert_path, server_key_path)`
    """
    cert_dir.mkdir(parents=True, exist_ok=True)
    ca_cert_path = cert_dir / "agy-root-ca.pem"
    ca_key_path = cert_dir / "agy-root-ca.key"
    server_cert_path = cert_dir / "server.crt"
    server_key_path = cert_dir / "server.key"
    ip_marker_path = cert_dir / "lan_ip.txt"

    # Regenerate server cert if LAN IP changed or certs don't exist yet
    if (
        ca_cert_path.exists()
        and server_cert_path.exists()
        and server_key_path.exists()
        and ip_marker_path.exists()
        and ip_marker_path.read_text(encoding="utf-8").strip() == lan_ip
    ):
        return ca_cert_path, server_cert_path, server_key_path

    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Create or load Root CA
    if ca_cert_path.exists() and ca_key_path.exists():
        ca_key = serialization.load_pem_private_key(
            ca_key_path.read_bytes(), password=None
        )
        ca_cert = x509.load_pem_x509_certificate(ca_cert_path.read_bytes())
    else:
        ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        ca_subject = x509.Name(
            [
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Antigravity Home Server"),
                x509.NameAttribute(
                    NameOID.COMMON_NAME, "Antigravity Home Bridge Local Root CA"
                ),
            ]
        )
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_subject)
            .issuer_name(ca_subject)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=3650))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=None), critical=True
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=True,
                    crl_sign=True,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(ca_key, hashes.SHA256())
        )
        ca_key_path.write_bytes(
            ca_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        ca_cert_path.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))

    # 2. Generate Server Leaf Certificate signed by Root CA with SAN for LAN IP
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_subject = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Antigravity Home Bridge"),
            x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
        ]
    )

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.DNSName(hostname),
        x509.DNSName("agy-home-bridge"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    except ValueError:
        san_entries.append(x509.DNSName(lan_ip))

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))  # Apple iOS max validity is 825 days
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(san_entries),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())  # type: ignore[arg-type]
    )

    server_key_path.write_bytes(
        server_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    server_cert_path.write_bytes(
        server_cert.public_bytes(serialization.Encoding.PEM)
        + ca_cert.public_bytes(serialization.Encoding.PEM)
    )
    ip_marker_path.write_text(lan_ip, encoding="utf-8")

    logger.info(
        "Generated LAN TLS certificate for https://%s (signed by %s)",
        lan_ip,
        ca_cert_path,
    )
    return ca_cert_path, server_cert_path, server_key_path
