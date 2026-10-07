from frontend.admin.metrics import group_users, user_metrics


def _user(status, role="USER"):
    return {"username": f"{status.lower()}-{role}", "role": role, "status": status}


def test_rejected_users_are_not_counted_as_users():
    users = [_user("ACTIVE")] * 8 + [_user("REJECTED"), _user("ACTIVE", role="ADMIN")]

    assert user_metrics(users) == {"users": 8, "active": 8, "pending": 0, "rejected": 1}


def test_active_and_pending_are_users():
    users = [_user("ACTIVE"), _user("ACTIVE"), _user("PENDING"), _user("REJECTED")]

    metrics = user_metrics(users)

    assert metrics["users"] == 3
    assert metrics["active"] == 2
    assert metrics["pending"] == 1


def test_admins_are_left_out_of_every_group():
    groups = group_users([_user("ACTIVE", role="ADMIN"), _user("PENDING")])

    assert groups["ACTIVE"] == []
    assert [item["status"] for item in groups["PENDING"]] == ["PENDING"]
