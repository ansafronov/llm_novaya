"""Render editorial review data as Markdown."""
import argparse
import re
from difflib import SequenceMatcher
from pathlib import Path

from .data import read_rows


QUALIFIERS = {
    'приблизительность': r'\b(?:примерно|приблизительно|около|порядка)\b',
    'нижняя граница': r'\b(?:как минимум|не менее|по меньшей мере|минимум)\b',
    'большинство': r'\b(?:большинство|большая часть|основная часть)\b',
    'наилучший случай': r'\bв лучшем случае\b',
    'неподтверждённость': r'\b(?:неподтвержд[её]н\w*|не подтвержд[её]н\w*|без подтверждения)\b',
}


def longest_overlap(text, reference):
    words = re.findall(r'\w+', text)
    reference_words = re.findall(r'\w+', reference.casefold())
    match = SequenceMatcher(
        None, [word.casefold() for word in words], reference_words, autojunk=False,
    ).find_longest_match()
    return ' '.join(words[match.a:match.a + match.size]) if match.size >= 8 else ''


def missing_years(text, evidence):
    pattern = r'(?<!\w)(?:18|19|20)\d{2}(?!\w)'

    def years(value):
        prose = re.sub(r'\]\([^\n)]*\)|https?://[^\s)>]+', '', value)
        return set(re.findall(pattern, prose))

    source_years = set().union(*(years(sentence) for sentence in evidence))
    return sorted(source_years - years(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot', type=Path)
    args = parser.parse_args()
    pairs = read_rows(args.pilot / 'review.jsonl')
    audits = read_rows(args.pilot / 'audit.jsonl')
    lines = ['# Проверка пилота', '',
             'Для каждой пары проверьте пересказ, атрибуцию, полноту фактов и копирование.']
    for index, pair in enumerate(pairs, 1):
        lines.extend(['', f'## Пара {index}', '', pair['source']])
        context = pair.get('context', {})
        if any(context.values()):
            lines.extend(['', '### Соседний контекст', ''])
            for key, label in [('title', 'Заголовок'), ('genre', 'Жанр'),
                               ('publication_date', 'Дата публикации'),
                               ('preceding', 'Предыдущий абзац'), ('following', 'Следующий абзац')]:
                if context.get(key):
                    lines.extend([f'**{label}**', '', context[key], ''])
        lines.extend(['', '### Карточка', ''])
        fragment = pair['fragment']['messages'][-1]['content']
        for fact in pair['facts']:
            lines.append('- ' + fact['text'])
            evidence = [pair['evidence'][source] for source in fact['sources']]
            for source in fact['sources']:
                lines.append(f"  - {source}: {pair['evidence'][source]}")
            missing = [label for label, pattern in QUALIFIERS.items()
                       if any(re.search(pattern, part, re.I) for part in evidence)
                       and not re.search(pattern, fact['text'], re.I)]
            if missing:
                lines.append('  - Проверьте, сохранены ли оговорки — ' + ', '.join(missing) + '.')
            years = missing_years(fact['text'], evidence)
            if years:
                lines.append('  - Проверьте, существенны ли пропущенные годы — ' + ', '.join(years) + '.')
            overlap = longest_overlap(fact['text'], fragment)
            if overlap:
                lines.append(f'  - Проверьте дословное совпадение — «{overlap}».')
        for key, label in [('fragment', 'Авторский фрагмент'), ('lead', 'Синтетический лид')]:
            example = pair[key]
            lines.extend(['', f'### {label}', '',
                          example['messages'][-1]['content'] if example else 'Не прошёл фильтр.',
                          '', '- [ ] Пара пригодна'])
    lines.extend(['', '## Что удалил разметчик', ''])
    for audit in audits:
        if audit.get('removed'):
            lines.extend([f"### {audit['source']}", ''])
            for item in audit['removed']:
                lines.extend([f"**Абзац {item['id']}: {item['reason']}**", '', item['text'],
                              '', '- [ ] Удалён содержательный текст', ''])
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
