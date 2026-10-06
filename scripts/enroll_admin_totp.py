"""Enroll an administrator account in TOTP two-factor authentication.

ADMIN accounts cannot log in until they have TOTP enabled. This script
generates a secret, stores it on the user and prints the otpauth:// URI to
scan with an authenticator app. It writes to the database only with --apply.

    uv run python -m scripts.enroll_admin_totp --username admin            # preview
    uv run python -m scripts.enroll_admin_totp --username admin --apply    # write
"""
from __future__ import annotations

import argparse

import pyotp
from sqlalchemy import select
from sqlalchemy.orm import Session

from auth.security import generate_totp_secret
from database import DatabaseSettings, create_db_engine
from database.models import User


ISSUER = "LawChat"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", required=True)
    parser.add_argument(
        "--apply", action="store_true", help="write the new secret to the database"
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="replace an existing secret (invalidates the current authenticator entry)",
    )
    args = parser.parse_args()

    engine = create_db_engine(DatabaseSettings.from_env())
    with Session(engine) as db, db.begin():
        user = db.scalar(select(User).where(User.username == args.username))
        if user is None:
            raise SystemExit(f"User {args.username!r} not found")
        if user.role != "ADMIN":
            raise SystemExit(f"User {args.username!r} has role {user.role}, not ADMIN")
        if user.totp_enabled and user.totp_secret and not args.rotate:
            raise SystemExit(
                f"User {args.username!r} already has TOTP enabled; pass --rotate to replace it"
            )
        if not args.apply:
            print(f"Would enroll {args.username!r} in TOTP. Rerun with --apply to write.")
            return

        secret = generate_totp_secret()
        user.totp_secret = secret
        user.totp_enabled = True
        user.totp_last_step = None

    uri = pyotp.TOTP(secret).provisioning_uri(name=args.username, issuer_name=ISSUER)
    print("TOTP enabled. Scan this URI with an authenticator app, then keep it secret:")
    print(uri)
    print("For scripts/admin_session.py set LAWCHAT_ADMIN_TOTP_SECRET to:")
    print(secret)


if __name__ == "__main__":
    main()
