import pytest

from lawchat.database import provision_key


def test_provision_key_is_normalized_and_hierarchical():
    assert provision_key(" 143 ", " 3 ", " A ") == "article:143/clause:3/point:a"
    assert provision_key("5") == "article:5/clause:/point:"
    with pytest.raises(ValueError):
        provision_key(None)
