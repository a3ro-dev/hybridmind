with open('D:/hybridmind/engine/hybrid_ranker.py', encoding='utf-8') as f:
    content = f.read()

# Reduce BM25 boost weight from 0.45 to 0.25 to prevent score cap at 1.0
# so that vector scores can differentiate results even with high keyword overlap
content = content.replace(
    'bm25_boost_weight: float = 0.45',
    'bm25_boost_weight: float = 0.25'
)

with open('D:/hybridmind/engine/hybrid_ranker.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('BM25_BOOST_REDUCED')
