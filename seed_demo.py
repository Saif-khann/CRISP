"""
CRISP - demo account seeding.

Creates two real accounts (hashed passwords, same login path as everyone
else) with a few projects and validation history, so the one-click demo
buttons have something to show. Idempotent: existing accounts are left
alone, so re-running this never duplicates or resets real activity.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

import database

logger = logging.getLogger(__name__)

# Passwords are overridable so a public deployment doesn't have to run
# with the documented default.
DEMO_ACCOUNTS = {
    'expert': {
        'email': os.getenv('DEMO_EXPERT_EMAIL', 'demo.expert@crisp.local'),
        'password': os.getenv('DEMO_EXPERT_PASSWORD', 'crisp-demo-expert'),
        'display_name': 'Demo Expert',
    },
    'worker': {
        'email': os.getenv('DEMO_WORKER_EMAIL', 'demo.worker@crisp.local'),
        'password': os.getenv('DEMO_WORKER_PASSWORD', 'crisp-demo-worker'),
        'display_name': 'Demo Worker',
    },
}

_DEMO_PROJECTS = [
    {
        'name': 'Metropolitan Commercial Plaza',
        'description': '32-storey mixed-use commercial tower with 3 basement levels.',
        'location': 'Plot 104, Financial District',
        'start_date': '2024-08-12',
        'end_date': '2026-04-30',
        'latitude': 28.6139,
        'longitude': 77.2090,
        'history': [
            ('foundation', 'Excavation', 94.5, 87.1, 30),
            ('foundation', 'concrete_pouring', 96.2, 85.4, 8),
        ],
    },
    {
        'name': 'Skyline Horizon Residences',
        'description': 'Twin residential towers with podium parking and rooftop amenities.',
        'location': 'Sector 62, City Center',
        'start_date': '2024-03-04',
        'end_date': '2025-11-20',
        'latitude': 19.0760,
        'longitude': 72.8777,
        'history': [
            ('foundation', 'concrete curing', 93.8, 86.0, 120),
            ('superstructure', 'Structural_Frame_Erection_(framing)', 98.8, 72.3, 20),
        ],
    },
    {
        'name': 'Greenfield Medical Complex',
        'description': 'Multi-specialty 500-bed hospital with a diagnostics wing.',
        'location': 'Health City Campus',
        'start_date': '2023-11-06',
        'end_date': '2025-07-18',
        'latitude': 12.9716,
        'longitude': 77.5946,
        'history': [
            ('superstructure', 'Roof_Decking', 91.4, 74.0, 90),
            ('facade', 'Window_and_Door_Installation', 96.8, 74.2, 14),
        ],
    },
]

_DESCRIPTIONS = {
    'Excavation': 'Site excavation underway with earthmoving plant active across the central plot.',
    'concrete_pouring': 'Concrete placement in progress with pump equipment positioned on site.',
    'concrete curing': 'Foundation curing phase with moisture retention measures in place.',
    'Structural_Frame_Erection_(framing)': 'Structural frame erection advancing; columns and deck framing visible.',
    'Roof_Decking': 'Upper roof decking and slab formwork assembly in progress.',
    'Window_and_Door_Installation': 'Glazing and window frame anchoring visible across the facade.',
}


def _sample_image_for(stage):
    """Point demo validations at the bundled sample photo for that stage,
    when it exists, so their PDF reports contain a real image."""
    mapping = {
        'foundation': 'sample_foundation.jpg',
        'superstructure': 'sample_superstructure.jpg',
        'facade': 'sample_facade.jpg',
        'Interior': 'sample_interior.jpg',
        'finishing works': 'sample_finishing.jpg',
    }
    filename = mapping.get(stage)
    if not filename:
        return None
    path = os.path.join('static', 'demo_samples', filename)
    return path if os.path.exists(path) else None


def _backdate(validation_id, days_ago):
    """Give seeded validations a believable spread of dates, so the
    milestone comparison feature has a real timeline to work with."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with database.get_connection() as conn:
        conn.execute(
            'UPDATE validations SET timestamp = ? WHERE id = ?', (ts, validation_id)
        )


def seed_demo_data():
    """Create demo accounts and their sample projects if absent."""
    for role, account in DEMO_ACCOUNTS.items():
        existing = database.get_user_by_email(account['email'])
        if existing:
            continue

        try:
            user_id = database.create_user(
                email=account['email'],
                password=account['password'],
                role=role,
                display_name=account['display_name'],
                is_demo=True,
            )
        except ValueError as exc:
            logger.warning("Could not seed %s demo account: %s", role, exc)
            continue

        for spec in _DEMO_PROJECTS:
            project_id = database.create_project(
                user_id=user_id,
                name=spec['name'],
                description=spec['description'],
                location=spec['location'],
                start_date=spec['start_date'],
                end_date=spec['end_date'],
                latitude=spec['latitude'],
                longitude=spec['longitude'],
            )

            last_stage = last_sub = None
            for stage, sub_stage, stage_conf, global_conf, days_ago in spec['history']:
                validation_id = database.create_validation(
                    user_id=user_id,
                    project_id=project_id,
                    primary_stage=stage,
                    specific_classification=sub_stage,
                    stage_confidence=stage_conf,
                    global_confidence=global_conf,
                    image_path=_sample_image_for(stage),
                    ai_description=_DESCRIPTIONS.get(sub_stage),
                )
                _backdate(validation_id, days_ago)
                last_stage, last_sub = stage, sub_stage

            if last_stage:
                database.update_project_stage(user_id, project_id, last_stage, last_sub)

        logger.info("Seeded %s demo account: %s", role, account['email'])
