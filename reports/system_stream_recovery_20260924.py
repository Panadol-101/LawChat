import json
from pathlib import Path
import httpx

prior = json.loads(Path('reports/system_chat_20260924.json').read_text())
cid = prior['conversation_id']
payload = {'conversation_id': cid, 'client_message_id': 'assessment-sse-recovery',
           'message': 'Điều 8 Luật Hôn nhân và gia đình 52/2014/QH13 quy định điều kiện kết hôn như thế nào?',
           'as_of': '2026-07-31'}
with httpx.Client(base_url='http://127.0.0.1:8000', timeout=330,
                  headers={'X-Workspace-ID': 'assessment-20260924'}) as client:
    response = client.post('/api/v1/chat/stream', json=payload)
    replay = client.post('/api/v1/chat', json=payload)
    messages = client.get('/api/v1/conversations/'+cid+'/messages').json()
report = {'http_status': response.status_code, 'sse_body': response.text,
          'replay_status': replay.status_code, 'replay': replay.json(), 'messages': messages}
Path('reports/system_stream_recovery_20260924.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({'http_status': response.status_code,
    'events': [line for line in response.text.splitlines() if line.startswith('event:')],
    'replay_status': replay.status_code, 'message_count': len(messages)},ensure_ascii=False))
