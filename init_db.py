"""
CRISP - database initialisation utility.

Creates the SQLite schema and seeds the demo accounts. The app also does
this automatically on boot; this script exists so you can prepare or
inspect the database without starting the server.

    python init_db.py              # create schema + seed demo accounts
    python init_db.py --clean-uploads   # also clear static/uploads
"""

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import database
from seed_demo import seed_demo_data, DEMO_ACCOUNTS


def clean_uploads_directory():
    """Remove uploaded images. Destructive - only runs when asked for."""
    uploads_dir = os.path.join('static', 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"Created {uploads_dir}")
        return

    removed = 0
    for filename in os.listdir(uploads_dir):
        if filename == '.gitkeep':
            continue
        path = os.path.join(uploads_dir, filename)
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
    print(f"Removed {removed} file(s) from {uploads_dir}")


def main():
    parser = argparse.ArgumentParser(description="Initialise the CRISP database.")
    parser.add_argument(
        '--clean-uploads', action='store_true',
        help='Also delete everything in static/uploads (destructive).'
    )
    args = parser.parse_args()

    print("Initialising CRISP database...")
    database.init_db()
    print(f"  Schema ready at {database.DB_PATH}")

    seed_demo_data()
    for role, account in DEMO_ACCOUNTS.items():
        user = database.get_user_by_email(account['email'])
        state = 'ready' if user else 'FAILED'
        print(f"  Demo {role} account ({account['email']}): {state}")

    if args.clean_uploads:
        clean_uploads_directory()

    print("\nDone.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
