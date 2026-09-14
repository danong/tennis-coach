import json

import pytest

from serve_review.workflow.collections import (
    CollectionStoreError,
    add_attempt,
    add_source,
    create_collection,
    delete_collection,
    list_collections,
    load_collection,
    remove_attempt,
    remove_source,
    rename_collection,
    resolve_collection,
    set_collection_tags,
)
from serve_review.workflow.records import CollectionRecord, WorkflowRecordError
from serve_review.workflow.workspace import WorkspacePaths

CID = "collection-0123456789abcdef"
SID = "source-0123456789abcdef"
AID = "attempt-0123456789abcdef01234567"


def test_record_is_strict_and_deterministic():
    record = CollectionRecord(1, CID, "Set", (SID,), (AID,), ("a", "z"))
    assert record.to_dict()["source_ids"] == [SID]
    assert record.to_json() == record.to_json()
    with pytest.raises(WorkflowRecordError):
        CollectionRecord.from_dict({**record.to_dict(), "extra": 1})
    with pytest.raises(WorkflowRecordError):
        CollectionRecord.from_dict({**record.to_dict(), "tags": "a"})
    with pytest.raises(WorkflowRecordError):
        CollectionRecord.from_dict({**record.to_dict(), "attempt_ids": [AID, AID]})
    with pytest.raises(WorkflowRecordError):
        CollectionRecord.from_json('{"schema_version":1,"schema_version":1,"collection_id":"%s","display_name":"x","source_ids":[],"attempt_ids":[],"tags":[]}' % CID)
    with pytest.raises(WorkflowRecordError):
        CollectionRecord.from_json('{"schema_version":1,"collection_id":"%s","display_name":"x","source_ids":[],"attempt_ids":[],"tags":[],"x":NaN}' % CID)


def test_crud_order_resolve_and_nonexclusive_membership(tmp_path, monkeypatch):
    workspace = WorkspacePaths(tmp_path)
    monkeypatch.setattr("serve_review.workflow.collections.load_source", lambda ws, sid: object())
    ids = iter((CID, "collection-fedcba9876543210"))
    first = create_collection(workspace, "Group", id_factory=lambda: next(ids),
                              source_ids=(SID,), attempt_ids=(AID,), tags=("z", "a"))
    second = create_collection(workspace, "Other", id_factory=lambda: next(ids), source_ids=(SID,), attempt_ids=(AID,))
    assert [r.collection_id for r in list_collections(workspace)] == [CID, second.collection_id]
    assert resolve_collection(workspace, "Group") == first
    with pytest.raises(CollectionStoreError):
        add_source(workspace, CID, SID)
    sid2 = "source-fedcba9876543210"
    add_source(workspace, CID, sid2)
    remove_source(workspace, CID, sid2)
    aid2 = "attempt-fedcba9876543210fedcba98"
    add_attempt(workspace, CID, aid2)
    remove_attempt(workspace, CID, aid2)
    assert rename_collection(workspace, CID, "Renamed").display_name == "Renamed"
    assert set_collection_tags(workspace, CID, ("b", "a")).tags == ("a", "b")
    source_manifest = tmp_path / "sources" / "keep"
    source_manifest.parent.mkdir()
    source_manifest.write_text("keep")
    delete_collection(workspace, CID)
    assert source_manifest.read_text() == "keep"
    assert load_collection(workspace, second.collection_id) == second


def test_malformed_entry_and_collision(tmp_path):
    workspace = WorkspacePaths(tmp_path)
    create_collection(workspace, "x", id_factory=lambda: CID)
    with pytest.raises(CollectionStoreError):
        create_collection(workspace, "y", id_factory=lambda: CID)
    path = workspace.collections
    (path / "junk.json").write_text(json.dumps({}))
    with pytest.raises(CollectionStoreError):
        list_collections(workspace)
