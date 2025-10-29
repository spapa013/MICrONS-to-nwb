import os

from caveclient import CAVEclient
from caveclient.base import AuthException


def get_client():
    try:
        client = CAVEclient("minnie65_public")
    except AuthException:
        # Initialize client without datastack name
        client = CAVEclient()
        # Access token from environment
        assert "CAVE" in os.environ, (
            "The token for CAVE could not be read from the environment."
            "Please set the environment variable named CAVE with the token."
        )
        token = os.environ["CAVE"]
        # Save token at default location used by caveclient
        client.auth.save_token(token=token, overwrite=True)

        # Retry access datastack
        client = CAVEclient("minnie65_public")
    return client


def get_functional_coreg_table(field_key):
    client = get_client()
    materialization_ver = client.materialize.version

    coreg_table = client.materialize.query_table(
        table="coregistration_manual_v4",
        split_positions=True,
    )
    session = field_key["session"]
    scan = field_key["scan_idx"]
    field = field_key["field"]

    coreg_table_for_this_field = coreg_table[
        (coreg_table["session"] == int(session)) & (coreg_table["scan_idx"] == int(scan)) & (coreg_table["field"] == int(field))
    ]

    return materialization_ver, coreg_table_for_this_field
