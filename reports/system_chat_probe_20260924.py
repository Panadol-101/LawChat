"""Persistence/SSE probe; deliberately retains assessment records."""
import json
from pathlib import Path
import httpx

base = 'http://127.0.0.1:8000'
headers = {'X-Workspace-ID': 'assessment-20260924'}
report = {}
with httpx.Client(base_url=base, headers=headers, timeout=330) as client:
    project = client.post('/api/v1/projects', json={'name': 'System assessment 2026-09-24'})
    report['create_project_status'] = project.status_code
    project.raise_for_status()
    conversation = client.post('/api/v1/conversations', json={
        'title': 'Retained assessment conversation', 'project_id': project.json()['id']})
    report['create_conversation_status'] = conversation.status_code
    conversation.raise_for_status()
    cid = conversation.json()['id']
    report['conversation_id'] = cid
    report['different_workspace_status'] = client.get('/api/v1/conversations/'+cid,
        headers={'X-Workspace-ID': 'assessment-other-20260924'}).status_code
    payload = {'conversation_id': cid, 'client_message_id': 'assessment-sse-1',
        'message': 'Hãy dự đoán chắc chắn tôi có thắng kiện không?', 'as_of': '2026-07-31'}
    stream = client.post('/api/v1/chat/stream', json=payload)
    report['sse_status'] = stream.status_code
    report['sse_content_type'] = stream.headers.get('content-type')
    report['sse_body'] = stream.text
    messages = client.get('/api/v1/conversations/'+cid+'/messages')
    report['messages_status'] = messages.status_code
    report['messages'] = messages.json()
    replay = client.post('/api/v1/chat', json=payload)
    report['replay_status'] = replay.status_code
    report['replay_body'] = replay.json()
    report['message_count_after_replay'] = len(client.get('/api/v1/conversations/'+cid+'/messages').json())
Path('reports/system_chat_20260924.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {'messages','sse_body','replay_body'}},ensure_ascii=False))
