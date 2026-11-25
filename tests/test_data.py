import math
import unittest
from novaya.data import parse_article, split_rows, encode, perplexity, evenly, temporal_sample


class Tokenizer:
    unk_token_id = -1
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return [10, 11, 12]
    def encode(self, text, **kwargs):
        return list(range(20, 20+len(text)))
    def convert_tokens_to_ids(self, token):
        return 99


class DataTests(unittest.TestCase):
    def test_cleaning(self):
        raw = '---\ntitle: Заголовок\ndate: "2024-01-02T10:00:00Z"\nsubtitle: секрет\n---\n# Заголовок\n\n![фото](image.jpg)\n\n'+('Слово '*60)+'\n\n## Раздел\nТекст\n**Читайте также:** [другой материал](relative.md)\n\n**Читайте также:**\n\nАвтор: [ссылка](related.md)\n\nСодержательный абзац\n**Читайте также:** https://example.org/related\n> **Читайте также:**\nВидео: [вставка](https://example.org/video)'
        row = parse_article(raw.replace('секрет', 'секрет\x97'), 'a.md')
        self.assertNotIn('image.jpg', row['text'])
        self.assertNotIn('relative.md', row['text'])
        self.assertNotIn('related.md', row['text'])
        self.assertNotIn('https://example.org/video', row['text'])
        self.assertNotIn('Читайте также', row['text'])
        self.assertIn('Содержательный абзац', row['text'])
        self.assertNotIn('Заголовок', row['text'])
        self.assertNotIn('секрет', row['text'])
        self.assertIn('## Раздел', row['text'])
        repaired = parse_article(raw.replace('Заголовок','Мой\x97заголовок'), 'control.md')
        self.assertEqual(repaired['title'], 'Мой—заголовок')
        self.assertNotIn('Мой—заголовок', repaired['text'])
        with self.assertRaisesRegex(ValueError, 'invalid_date'):
            parse_article(raw.replace('2024-01-02T10:00:00Z','bad'), 'a')

    def test_dedup_and_split(self):
        rows = [dict(date=f'2024-01-{i:02d}', source=str(i), id=str(i)) for i in range(1,21)]
        rows += [dict(date='2024-02-01', source='duplicate', id='1')]
        train, val, duplicates = split_rows(list(reversed(rows)))
        self.assertEqual(duplicates,1)
        self.assertEqual(len(val),2)
        self.assertLess(max(r['date'] for r in train), min(r['date'] for r in val))
        self.assertFalse(set(r['id'] for r in train)&set(r['id'] for r in val))
        self.assertEqual(train[0]['date'],'2024-01-01')

    def test_earliest_timestamp_wins_within_day(self):
        rows = [dict(date='2024-01-01', published_at='2024-01-01T12:00:00+00:00',source='a',id='same'),
                dict(date='2024-01-01', published_at='2024-01-01T01:00:00+00:00',source='z',id='same'),
                dict(date='2024-01-02', published_at='2024-01-02T00:00:00+00:00',source='b',id='other')]
        train, val, removed = split_rows(rows)
        self.assertEqual(train[0]['source'],'z')
        self.assertEqual(removed,1)

    def test_mask_and_truncation(self):
        row = {'title':'T','text':'abc'}
        full = encode(Tokenizer(), row, 20)
        self.assertEqual(full['labels'],[-100,-100,-100,20,21,22,99])
        cut = encode(Tokenizer(), row, 5)
        self.assertEqual(cut['labels'],[-100,-100,-100,20,21])
        self.assertTrue(cut['truncated'])
        self.assertNotIn(99,cut['input_ids'])

    def test_calendar_sampling(self):
        rows = [dict(date='2024-01-01', id=str(i)) for i in range(100)]
        rows += [dict(date='2024-06-30',id='middle'),dict(date='2024-12-31',id='end')]
        sample = temporal_sample(rows,3)
        self.assertEqual([r['date'] for r in sample], ['2024-01-01','2024-06-30','2024-12-31'])
        self.assertEqual(len({r['id'] for r in temporal_sample(rows,50)}),50)

    def test_ppl_and_selection(self):
        self.assertAlmostEqual(perplexity(2*math.log(2)+8*math.log(4),10),2**1.8)
        with self.assertRaises(ValueError):
            perplexity(0,0)
        self.assertEqual(evenly(list(range(1000)), 3), [0,500,999])


if __name__ == '__main__':
    unittest.main()
