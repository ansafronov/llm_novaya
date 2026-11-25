import tempfile
import unittest
from pathlib import Path

from novaya.data import file_sha256, write_json, write_rows
from novaya.dialogues import approved_rows, candidates, fingerprint


def message(message_id, role, parent=None, **changes):
    return dict({
        'message_id': message_id, 'message_tree_id': 'tree', 'parent_id': parent,
        'role': role, 'text': f'Реплика {message_id}', 'lang': 'ru',
        'review_result': True, 'synthetic': False, 'deleted': False, 'rank': 0,
    }, **changes)


class DialogueTests(unittest.TestCase):
    def test_keeps_one_complete_reviewed_branch(self):
        rows = [
            message('root', 'prompter'),
            message('answer', 'assistant', 'root'),
            message('followup', 'prompter', 'answer'),
            message('wrong', 'assistant', 'followup', rank=1),
            message('ending', 'assistant', 'followup'),
            message('unreviewed', 'prompter', 'ending', review_result=False),
            message('orphan', 'assistant', 'unreviewed'),
            message('flagged', 'assistant', 'followup', labels={'name': ['pii'], 'value': [0.5]}),
        ]
        result, = candidates(reversed(rows), lambda branch: len(branch) <= 4)
        self.assertEqual(result['message_ids'], ['root', 'answer', 'followup', 'ending'])
        self.assertEqual([row['role'] for row in result['messages']],
                         ['user', 'assistant', 'user', 'assistant'])

    def test_review_cannot_approve_modified_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            fingerprints = bundle / 'train.fingerprints.json'
            review = bundle / 'train.review.jsonl'
            candidate = dict(id='tree', message_ids=['root', 'answer'],
                             messages=[dict(role='user', content='Привет'),
                                       dict(role='assistant', content='Здравствуйте')])
            write_json(fingerprints, {candidate['id']: fingerprint(candidate)})
            manifest = {'splits': {'train': {'fingerprints_sha256': file_sha256(fingerprints)}}}
            write_rows(review, [{**candidate, 'approved': None}])
            self.assertEqual(approved_rows(bundle, 'train', manifest), [])
            write_rows(review, [{**candidate, 'approved': True}])
            self.assertEqual(approved_rows(bundle, 'train', manifest), [candidate])
            candidate['messages'][-1]['content'] = 'Другая реплика'
            write_rows(review, [{**candidate, 'approved': True}])
            with self.assertRaisesRegex(ValueError, 'changed review candidate'):
                approved_rows(bundle, 'train', manifest)
