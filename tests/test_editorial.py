import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from novaya import editorial
from novaya.editorial_style import article_chunks, clean_paragraph, eligible_spans
from novaya.data import write_rows
from novaya.editorial_training import editorial_rows


class EditorialTest(unittest.TestCase):
    def test_style_cleanup_preserves_author_prose_and_link_labels(self):
        text = '## Подзаголовок\n> **Они** [сказали](https://example.org/a_(b)): «Да».'
        self.assertEqual('Они сказали: «Да».', clean_paragraph(text, 'Заголовок'))
        self.assertEqual('', clean_paragraph('Заголовок', 'Заголовок'))
        self.assertEqual('', clean_paragraph('Фото: архив', 'Заголовок'))
        self.assertEqual('Он сказал: «порядок без ***».',
                         clean_paragraph('**Он** сказал: «порядок без ***».', 'Заголовок'))

    def test_style_window_does_not_join_across_removed_paragraph(self):
        self.assertEqual(set(), eligible_spans({1: 'слово ' * 20, 3: 'слово ' * 20}))

    def test_style_article_chunks_preserve_order_without_overlap(self):
        paragraphs = {1: 'abc', 3: 'def', 4: 'ghi'}
        self.assertEqual([paragraphs], list(article_chunks(paragraphs, 9)))
        self.assertEqual([{1: 'abc', 3: 'def'}, {4: 'ghi'}], list(article_chunks(paragraphs, 6)))

    def test_standalone_media_credit(self):
        matches = editorial.standalone_media_credit
        self.assertTrue(matches('*Олег Навальный. Фото: пресс-служба суда*'))
        self.assertTrue(matches('*Скрин: YouTube*'))
        self.assertFalse(matches('Адвокат подал апелляцию. Фото: пресс-служба суда'))
        self.assertFalse(matches('*Олег Навальный. Суд изменил меру пресечения. Фото: пресс-служба суда*'))

    def test_sentence_boundaries_preserve_russian_initials_and_acting_title(self):
        text = 'Выступил и. о. президента. На приёме вспомнили А. С. Пушкина. Затем он ушёл.'
        self.assertEqual(
            ['Выступил и. о. президента.', 'На приёме вспомнили А. С. Пушкина.', 'Затем он ушёл.'],
            editorial.sentences(text),
        )

    def test_sentence_boundaries_preserve_thousands_abbreviation(self):
        self.assertEqual(
            ['На канал подписаны 23 тыс. человек.', 'Канал заблокирован.'],
            editorial.sentences('На канал подписаны 23 тыс. человек. Канал заблокирован.'),
        )

    def test_rejected_card_never_reaches_lead_generation(self):
        text = ' '.join(['Содержательное предложение с описанием разных событий.'] * 8)
        responses = {
            editorial.CLEAN: {'remove': []},
            editorial.FACTS: {'blocks': [{'ids': [1], 'anchors': ['1.1', '1.2', '1.3']}]},
            editorial.REWRITE: {'texts': ['Пересказ 1', 'Пересказ 2', 'Пересказ 3']},
            editorial.CHECK_CARD: {'errors': ['subject changed'], 'missing': []},
        }
        issues = []
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            (archive / 'article.md').write_text(text)
            with patch.object(editorial, 'parse_article', return_value={'text': text}):
                with patch.object(editorial, 'request', side_effect=(
                    lambda port, prompt, payload, validate: validate(responses[prompt])
                )):
                    pairs, _ = editorial.prepare(
                        {'id': 'article', 'source': 'article.md'}, archive, 8080, issues)
        self.assertEqual([], pairs)
        self.assertEqual('unsupported_or_incomplete_card', issues[0]['reason'])

    def test_failed_lead_and_invalid_block_preserve_other_fragment(self):
        text = ' '.join(['Содержательное предложение с описанием разных событий.'] * 8)

        def respond(port, prompt, payload, validate):
            if prompt == editorial.CLEAN:
                return validate({'remove': [{'id': 2, 'reason': 'caption'}]})
            if prompt == editorial.FACTS:
                return validate({'blocks': [
                    {'ids': [1], 'anchors': ['1.1', '1.2', '9.9']},
                    {'ids': [1], 'anchors': ['1.1', '1.2', '1.3']},
                ]})
            if prompt == editorial.REWRITE:
                return validate({'texts': ['Пересказ 1', 'Пересказ 2', 'Пересказ 3']})
            if prompt == editorial.CHECK_CARD:
                return validate({'errors': [], 'missing': []})
            return validate({'lead': 'Незаконченный лид'})

        issues = []
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)
            (archive / 'article.md').write_text(text)
            with patch.object(editorial, 'parse_article', return_value={'text': text + '\n\nФото'}):
                with patch.object(editorial, 'request', side_effect=respond):
                    pairs, removed = editorial.prepare(
                        {'id': 'article', 'source': 'article.md'}, archive, 8080, issues)
        self.assertEqual('Фото', removed[0]['text'])
        self.assertEqual(text, pairs[0]['fragment']['messages'][-1]['content'])
        self.assertIsNone(pairs[0]['lead'])
        self.assertEqual(dict(stage='facts', ids=[1], reason='invalid sentence anchors'), issues[0])
        self.assertEqual('lead', issues[1]['stage'])

    def test_paired_variants_share_order_and_source_fragments(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            author = editorial.example('Автор', 'Факт', 'Фрагмент')
            lead = editorial.example('Лид', 'Факт', 'Вводный текст')
            write_rows(directory / 'pairs.jsonl', [
                {'article_id': 'article', 'ids': [1], 'fragment': author, 'lead': lead},
                {'article_id': 'article', 'ids': [2], 'fragment': author, 'lead': None},
                {'article_id': 'article', 'ids': [3], 'fragment': author, 'lead': lead},
            ])
            for variant in ('lead', 'fragment'):
                self.assertEqual([[1], [3]], [row['ids'] for row in editorial_rows(directory, variant, True)])
