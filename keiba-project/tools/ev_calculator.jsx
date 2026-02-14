import { useState, useMemo, useCallback } from "react";

// 小倉牝馬S2026 sample data
const SAMPLE_HORSES = [
  { num: 1, name: "テレサ", score: 72, odds_win: 7.4, odds_place: 2.1 },
  { num: 2, name: "ブラウンラチェット", score: 45, odds_win: 46.6, odds_place: 9.0 },
  { num: 3, name: "フレミングフープ", score: 65, odds_win: 22.4, odds_place: 4.5 },
  { num: 4, name: "クリノメイ", score: 62, odds_win: 42.8, odds_place: 7.0 },
  { num: 5, name: "アレナリア", score: 35, odds_win: 75.9, odds_place: 15.0 },
  { num: 6, name: "フィールシンパシー", score: 40, odds_win: 51.1, odds_place: 10.0 },
  { num: 7, name: "インヴォーグ", score: 60, odds_win: 19.6, odds_place: 4.0 },
  { num: 8, name: "ココナッツブラウン", score: 80, odds_win: 4.7, odds_place: 1.6 },
  { num: 9, name: "パレハ", score: 66, odds_win: 17.5, odds_place: 3.8 },
  { num: 10, name: "タクシンイメル", score: 30, odds_win: 193.8, odds_place: 30.0 },
  { num: 11, name: "エリカヴィータ", score: 38, odds_win: 68.1, odds_place: 12.0 },
  { num: 12, name: "アンリーロード", score: 32, odds_win: 97.1, odds_place: 18.0 },
  { num: 13, name: "ウインエーデル", score: 33, odds_win: 103.0, odds_place: 18.0 },
  { num: 14, name: "クリスマスパレード", score: 68, odds_win: 15.1, odds_place: 3.5 },
  { num: 15, name: "レディーヴァリュー", score: 67, odds_win: 11.1, odds_place: 3.0 },
  { num: 16, name: "ボンドガール", score: 58, odds_win: 27.0, odds_place: 5.5 },
  { num: 17, name: "ジョスラン", score: 85, odds_win: 2.4, odds_place: 1.3 },
  { num: 18, name: "パルクリチュード", score: 34, odds_win: 71.0, odds_place: 13.0 },
];

// Actual result for validation
const ACTUAL_RESULT = { first: 17, second: 16, third: 8 };

// Softmax probability conversion
function softmax(scores, temperature) {
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - maxScore) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

// Calculate place probabilities (top 3) using conditional
function calcPlaceProbs(winProbs) {
  const n = winProbs.length;
  const placeProbs = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    // P(i in top 3) ≈ iterate through scenarios
    // Simplified: use win prob + adjusted second/third probs
    let pTop3 = 0;

    // P(i wins)
    pTop3 += winProbs[i];

    // P(i 2nd) = sum over j!=i: P(j wins) * P(i wins among rest) / (1 - P(j wins))
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const pIgivenNotJ = winProbs[i] / (1 - winProbs[j]);
      pTop3 += winProbs[j] * pIgivenNotJ;
    }

    // P(i 3rd) = sum over j,k!=i: P(j wins) * P(k 2nd|j wins) * P(i 3rd|j,k top2)
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      for (let k = 0; k < n; k++) {
        if (k === i || k === j) continue;
        const pK2nd = winProbs[k] / (1 - winProbs[j]);
        const pI3rd = winProbs[i] / (1 - winProbs[j] - winProbs[k]);
        pTop3 += winProbs[j] * pK2nd * pI3rd;
      }
    }

    placeProbs[i] = Math.min(pTop3, 1);
  }

  return placeProbs;
}

// Quinella (馬連) probability
function calcQuinellaProb(winProbs, i, j) {
  const pIJ = winProbs[i] * (winProbs[j] / (1 - winProbs[i]));
  const pJI = winProbs[j] * (winProbs[i] / (1 - winProbs[j]));
  return pIJ + pJI;
}

// Wide probability (both in top 3)
function calcWideProb(winProbs, i, j) {
  const n = winProbs.length;
  let prob = 0;

  // Both i and j in top 3
  // Case 1: i=1st, j in 2nd or 3rd
  // Case 2: j=1st, i in 2nd or 3rd
  // Case 3: k=1st (k!=i,j), i and j in 2nd,3rd

  // Case 1: i wins, j in top3-rest
  const pJ_given_I_wins = winProbs[j] / (1 - winProbs[i]);
  let pJ_top2_given_I_wins = pJ_given_I_wins; // j is 2nd
  for (let k = 0; k < n; k++) {
    if (k === i || k === j) continue;
    const pK2 = winProbs[k] / (1 - winProbs[i]);
    const pJ3 = winProbs[j] / (1 - winProbs[i] - winProbs[k]);
    pJ_top2_given_I_wins += pK2 * pJ3;
  }
  prob += winProbs[i] * pJ_top2_given_I_wins;

  // Case 2: j wins, i in top3-rest
  const pI_given_J_wins = winProbs[i] / (1 - winProbs[j]);
  let pI_top2_given_J_wins = pI_given_J_wins;
  for (let k = 0; k < n; k++) {
    if (k === i || k === j) continue;
    const pK2 = winProbs[k] / (1 - winProbs[j]);
    const pI3 = winProbs[i] / (1 - winProbs[j] - winProbs[k]);
    pI_top2_given_J_wins += pK2 * pI3;
  }
  prob += winProbs[j] * pI_top2_given_J_wins;

  // Case 3: someone else wins, both i and j in 2nd/3rd
  for (let k = 0; k < n; k++) {
    if (k === i || k === j) continue;
    const rest = 1 - winProbs[k];
    const pI2 = winProbs[i] / rest;
    const pJ3 = winProbs[j] / (rest - winProbs[i]);
    const pJ2 = winProbs[j] / rest;
    const pI3 = winProbs[i] / (rest - winProbs[j]);
    prob += winProbs[k] * (pI2 * pJ3 + pJ2 * pI3);
  }

  return Math.min(prob, 1);
}

// Trio (3連複) probability
function calcTrioProb(winProbs, i, j, k) {
  // All permutations of i,j,k in positions 1,2,3
  const perms = [
    [i, j, k], [i, k, j], [j, i, k],
    [j, k, i], [k, i, j], [k, j, i],
  ];

  let prob = 0;
  for (const [a, b, c] of perms) {
    const pA = winProbs[a];
    const pB = winProbs[b] / (1 - winProbs[a]);
    const pC = winProbs[c] / (1 - winProbs[a] - winProbs[b]);
    prob += pA * pB * pC;
  }

  return prob;
}

function formatPct(v) {
  return (v * 100).toFixed(1) + "%";
}
function formatYen(v) {
  return "¥" + Math.round(v).toLocaleString();
}

const TABS = [
  { id: "input", label: "評価点入力" },
  { id: "prob", label: "確率変換" },
  { id: "ev", label: "期待値計算" },
  { id: "validate", label: "検証" },
];

export default function EVCalculator() {
  const [horses, setHorses] = useState(SAMPLE_HORSES);
  const [temperature, setTemperature] = useState(8);
  const [budget, setBudget] = useState(3000);
  const [activeTab, setActiveTab] = useState("prob");
  const [betUnit, setBetUnit] = useState(100);
  const [topN, setTopN] = useState(6);

  const updateHorse = useCallback((idx, field, value) => {
    setHorses((prev) => {
      const next = [...prev];
      next[idx] = { ...next[idx], [field]: Number(value) || 0 };
      return next;
    });
  }, []);

  // Core calculations
  const calc = useMemo(() => {
    const scores = horses.map((h) => h.score);
    const winProbs = softmax(scores, temperature);
    const placeProbs = calcPlaceProbs(winProbs);

    // Sort by score descending
    const ranked = horses
      .map((h, i) => ({ ...h, idx: i, winProb: winProbs[i], placeProb: placeProbs[i] }))
      .sort((a, b) => b.score - a.score);

    // Top N horses for combination bets
    const topHorses = ranked.slice(0, topN);
    const topIndices = topHorses.map((h) => h.idx);

    // Quinella combinations
    const quinellas = [];
    for (let a = 0; a < topIndices.length; a++) {
      for (let b = a + 1; b < topIndices.length; b++) {
        const i = topIndices[a];
        const j = topIndices[b];
        const prob = calcQuinellaProb(winProbs, i, j);
        // Estimate quinella odds from win odds
        const estOdds = Math.max(
          (horses[i].odds_win * horses[j].odds_win) / 10,
          1.1
        );
        const ev = betUnit * estOdds * prob;
        quinellas.push({
          pair: `${horses[i].num}-${horses[j].num}`,
          nameA: horses[i].name,
          nameB: horses[j].name,
          prob,
          estOdds,
          ev,
          evRatio: ev / betUnit,
        });
      }
    }
    quinellas.sort((a, b) => b.ev - a.ev);

    // Wide combinations
    const wides = [];
    for (let a = 0; a < topIndices.length; a++) {
      for (let b = a + 1; b < topIndices.length; b++) {
        const i = topIndices[a];
        const j = topIndices[b];
        const prob = calcWideProb(winProbs, i, j);
        const estOdds = Math.max(
          (horses[i].odds_win * horses[j].odds_win) / 30,
          1.1
        );
        const ev = betUnit * estOdds * prob;
        wides.push({
          pair: `${horses[i].num}-${horses[j].num}`,
          nameA: horses[i].name,
          nameB: horses[j].name,
          prob,
          estOdds,
          ev,
          evRatio: ev / betUnit,
        });
      }
    }
    wides.sort((a, b) => b.ev - a.ev);

    // Trio combinations (top 5 only to limit)
    const trioIndices = topIndices.slice(0, Math.min(topN, 7));
    const trios = [];
    for (let a = 0; a < trioIndices.length; a++) {
      for (let b = a + 1; b < trioIndices.length; b++) {
        for (let c = b + 1; c < trioIndices.length; c++) {
          const i = trioIndices[a];
          const j = trioIndices[b];
          const k = trioIndices[c];
          const prob = calcTrioProb(winProbs, i, j, k);
          const estOdds = Math.max(
            (horses[i].odds_win * horses[j].odds_win * horses[k].odds_win) / 200,
            1.5
          );
          const ev = betUnit * estOdds * prob;
          trios.push({
            trio: `${horses[i].num}-${horses[j].num}-${horses[k].num}`,
            names: [horses[i].name, horses[j].name, horses[k].name],
            prob,
            estOdds,
            ev,
            evRatio: ev / betUnit,
          });
        }
      }
    }
    trios.sort((a, b) => b.ev - a.ev);

    // Win bets
    const winBets = ranked.map((h) => {
      const ev = betUnit * h.odds_win * h.winProb;
      return {
        ...h,
        ev,
        evRatio: ev / betUnit,
      };
    });

    // Place bets
    const placeBets = ranked.map((h) => {
      const ev = betUnit * h.odds_place * h.placeProb;
      return {
        ...h,
        ev,
        evRatio: ev / betUnit,
      };
    });

    // Race confidence score
    const maxWinProb = Math.max(...winProbs);
    const entropy = -winProbs.reduce(
      (s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0
    );
    const maxEntropy = Math.log2(horses.length);
    const confidence = 1 - entropy / maxEntropy;

    return {
      winProbs,
      placeProbs,
      ranked,
      quinellas,
      wides,
      trios,
      winBets,
      placeBets,
      confidence,
      entropy,
    };
  }, [horses, temperature, topN, betUnit]);

  // Validation against actual result
  const validation = useMemo(() => {
    const actual = ACTUAL_RESULT;
    const winnerIdx = horses.findIndex((h) => h.num === actual.first);
    const secondIdx = horses.findIndex((h) => h.num === actual.second);
    const thirdIdx = horses.findIndex((h) => h.num === actual.third);

    return {
      winnerProb: calc.winProbs[winnerIdx],
      winnerRank: calc.ranked.findIndex((h) => h.num === actual.first) + 1,
      secondProb: calc.winProbs[secondIdx],
      secondRank: calc.ranked.findIndex((h) => h.num === actual.second) + 1,
      thirdProb: calc.winProbs[thirdIdx],
      thirdRank: calc.ranked.findIndex((h) => h.num === actual.third) + 1,
      quinellaProb: calcQuinellaProb(calc.winProbs, winnerIdx, secondIdx),
      wideProbs: [
        calcWideProb(calc.winProbs, winnerIdx, secondIdx),
        calcWideProb(calc.winProbs, winnerIdx, thirdIdx),
        calcWideProb(calc.winProbs, secondIdx, thirdIdx),
      ],
      trioProb: calcTrioProb(calc.winProbs, winnerIdx, secondIdx, thirdIdx),
    };
  }, [calc, horses]);

  const s = {
    root: {
      fontFamily: "'Noto Sans JP', 'Hiragino Sans', sans-serif",
      background: "#0B0E11",
      color: "#D1D5DB",
      minHeight: "100vh",
      padding: "16px",
    },
    header: {
      maxWidth: 960, margin: "0 auto", paddingBottom: 16,
      borderBottom: "1px solid #1F2937",
    },
    title: { fontSize: 20, fontWeight: 700, color: "#F9FAFB", margin: 0 },
    subtitle: { fontSize: 12, color: "#6B7280", marginTop: 4 },
    main: { maxWidth: 960, margin: "0 auto", marginTop: 16 },
    tabs: {
      display: "flex", gap: 2, marginBottom: 16,
      borderBottom: "1px solid #1F2937", paddingBottom: 0,
    },
    tab: (active) => ({
      padding: "8px 16px", fontSize: 13, fontWeight: active ? 600 : 400,
      color: active ? "#60A5FA" : "#6B7280",
      background: "transparent", border: "none",
      borderBottom: active ? "2px solid #60A5FA" : "2px solid transparent",
      cursor: "pointer", transition: "all 0.15s",
    }),
    card: {
      background: "#111827", border: "1px solid #1F2937",
      borderRadius: 10, padding: 16, marginBottom: 12,
    },
    cardTitle: { fontSize: 14, fontWeight: 600, color: "#E5E7EB", marginBottom: 10 },
    table: { width: "100%", borderCollapse: "collapse", fontSize: 12 },
    th: {
      textAlign: "left", padding: "6px 8px", borderBottom: "1px solid #1F2937",
      color: "#9CA3AF", fontWeight: 500, fontSize: 11, whiteSpace: "nowrap",
    },
    td: (highlight) => ({
      padding: "5px 8px", borderBottom: "1px solid #111827",
      color: highlight ? "#F9FAFB" : "#D1D5DB",
      fontWeight: highlight ? 600 : 400,
    }),
    input: {
      width: 54, padding: "3px 6px", fontSize: 12,
      background: "#1F2937", border: "1px solid #374151",
      borderRadius: 4, color: "#F9FAFB", textAlign: "right",
    },
    slider: {
      width: "100%", accentColor: "#3B82F6",
    },
    badge: (color) => ({
      display: "inline-block", padding: "1px 6px", borderRadius: 3,
      fontSize: 10, fontWeight: 600,
      color: color, background: color + "18", border: `1px solid ${color}40`,
    }),
    evBar: (ratio) => ({
      height: 4, borderRadius: 2,
      background: ratio >= 1.2 ? "#22C55E" : ratio >= 1.0 ? "#EAB308" : "#6B7280",
      width: Math.min(ratio / 2 * 100, 100) + "%",
      transition: "width 0.3s",
    }),
  };

  const evBadge = (ratio) => {
    if (ratio >= 1.5) return { label: "S級", color: "#22C55E" };
    if (ratio >= 1.2) return { label: "A級", color: "#3B82F6" };
    if (ratio >= 1.0) return { label: "B級", color: "#EAB308" };
    return { label: "C級", color: "#6B7280" };
  };

  return (
    <div style={s.root}>
      <div style={s.header}>
        <h1 style={s.title}>馬券期待値計算機 v0.1</h1>
        <p style={s.subtitle}>
          評価点 → ソフトマックス確率変換 → 全券種期待値自動算出
        </p>
      </div>

      <div style={s.main}>
        {/* Tabs */}
        <div style={s.tabs}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              style={s.tab(activeTab === t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Controls bar */}
        <div style={{ ...s.card, display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 11, color: "#9CA3AF", marginBottom: 4 }}>
              温度パラメータ (T={temperature})
            </div>
            <input
              type="range" min="2" max="25" step="0.5"
              value={temperature} onChange={(e) => setTemperature(Number(e.target.value))}
              style={{ ...s.slider, width: 150 }}
            />
            <div style={{ fontSize: 10, color: "#6B7280", marginTop: 2 }}>
              低い=実力差拡大　高い=混戦
            </div>
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#9CA3AF", marginBottom: 4 }}>
              組合せ上位N頭
            </div>
            <input
              type="range" min="4" max="10" step="1"
              value={topN} onChange={(e) => setTopN(Number(e.target.value))}
              style={{ ...s.slider, width: 100 }}
            />
            <span style={{ fontSize: 12, color: "#E5E7EB", marginLeft: 8 }}>{topN}頭</span>
          </div>
          <div>
            <div style={{ fontSize: 11, color: "#9CA3AF", marginBottom: 4 }}>予算</div>
            <input
              type="number" value={budget} step={500}
              onChange={(e) => setBudget(Number(e.target.value))}
              style={{ ...s.input, width: 70 }}
            />
            <span style={{ fontSize: 11, color: "#6B7280", marginLeft: 4 }}>円</span>
          </div>
          <div style={{ marginLeft: "auto", textAlign: "right" }}>
            <div style={{ fontSize: 11, color: "#9CA3AF" }}>レース確信度</div>
            <div style={{
              fontSize: 22, fontWeight: 700,
              color: calc.confidence > 0.3 ? "#22C55E" : calc.confidence > 0.15 ? "#EAB308" : "#EF4444",
            }}>
              {(calc.confidence * 100).toFixed(0)}%
            </div>
            <div style={{ fontSize: 10, color: "#6B7280" }}>
              {calc.confidence > 0.3 ? "高確信・買い" : calc.confidence > 0.15 ? "中確信・標準" : "低確信・見送り検討"}
            </div>
          </div>
        </div>

        {/* INPUT TAB */}
        {activeTab === "input" && (
          <div style={s.card}>
            <div style={s.cardTitle}>評価点・オッズ入力（小倉牝馬S2026プリセット）</div>
            <div style={{ overflowX: "auto" }}>
              <table style={s.table}>
                <thead>
                  <tr>
                    <th style={s.th}>番</th>
                    <th style={s.th}>馬名</th>
                    <th style={s.th}>評価点</th>
                    <th style={s.th}>単勝</th>
                    <th style={s.th}>複勝</th>
                  </tr>
                </thead>
                <tbody>
                  {horses.map((h, i) => (
                    <tr key={h.num}>
                      <td style={s.td(false)}>{h.num}</td>
                      <td style={s.td(true)}>{h.name}</td>
                      <td style={s.td(false)}>
                        <input
                          style={s.input} value={h.score}
                          onChange={(e) => updateHorse(i, "score", e.target.value)}
                        />
                      </td>
                      <td style={s.td(false)}>
                        <input
                          style={s.input} value={h.odds_win}
                          onChange={(e) => updateHorse(i, "odds_win", e.target.value)}
                        />
                      </td>
                      <td style={s.td(false)}>
                        <input
                          style={s.input} value={h.odds_place}
                          onChange={(e) => updateHorse(i, "odds_place", e.target.value)}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* PROBABILITY TAB */}
        {activeTab === "prob" && (
          <>
            <div style={s.card}>
              <div style={s.cardTitle}>確率変換結果（ソフトマックス T={temperature}）</div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>順位</th>
                      <th style={s.th}>番</th>
                      <th style={s.th}>馬名</th>
                      <th style={s.th}>評価点</th>
                      <th style={s.th}>1着確率</th>
                      <th style={s.th}>3着内確率</th>
                      <th style={s.th}>確率バー</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calc.ranked.map((h, i) => (
                      <tr key={h.num} style={{
                        background: h.num === ACTUAL_RESULT.first ? "#22C55E12"
                          : h.num === ACTUAL_RESULT.second ? "#3B82F612"
                          : h.num === ACTUAL_RESULT.third ? "#EAB30812"
                          : "transparent",
                      }}>
                        <td style={s.td(false)}>{i + 1}</td>
                        <td style={s.td(false)}>{h.num}</td>
                        <td style={s.td(true)}>{h.name}</td>
                        <td style={s.td(false)}>{h.score}</td>
                        <td style={s.td(true)}>{formatPct(h.winProb)}</td>
                        <td style={s.td(true)}>{formatPct(h.placeProb)}</td>
                        <td style={{ ...s.td(false), width: 120 }}>
                          <div style={{
                            background: "#1F2937", borderRadius: 2, height: 6,
                          }}>
                            <div style={{
                              height: 6, borderRadius: 2,
                              background: `linear-gradient(90deg, #3B82F6, #60A5FA)`,
                              width: `${Math.min(h.winProb * 100 / Math.max(...calc.winProbs) * 100, 100)}%`,
                            }} />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ fontSize: 11, color: "#6B7280", marginTop: 8 }}>
                背景色: 🟢1着 🔵2着 🟡3着（実際の結果）
              </div>
            </div>

            <div style={s.card}>
              <div style={s.cardTitle}>温度パラメータの影響</div>
              <div style={{ fontSize: 12, color: "#9CA3AF", lineHeight: 1.7 }}>
                T={temperature} → 1位 {calc.ranked[0]?.name} の1着確率 = {formatPct(calc.ranked[0]?.winProb || 0)}
                {temperature < 5 && " ← 実力差を過大評価（堅い決着想定）"}
                {temperature > 15 && " ← 実力差を過小評価（大荒れ想定）"}
                {temperature >= 5 && temperature <= 15 && " ← 標準的な差の反映"}
              </div>
              <div style={{ fontSize: 11, color: "#6B7280", marginTop: 6 }}>
                目安: 堅いレース T=5-7 / 標準 T=8-12 / 荒れるレース T=13-20
              </div>
            </div>
          </>
        )}

        {/* EV TAB */}
        {activeTab === "ev" && (
          <>
            {/* Win EV */}
            <div style={s.card}>
              <div style={s.cardTitle}>単勝 期待値ランキング</div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>番</th>
                      <th style={s.th}>馬名</th>
                      <th style={s.th}>確率</th>
                      <th style={s.th}>オッズ</th>
                      <th style={s.th}>期待値</th>
                      <th style={s.th}>判定</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calc.winBets.slice(0, 8).map((h) => {
                      const b = evBadge(h.evRatio);
                      return (
                        <tr key={h.num}>
                          <td style={s.td(false)}>{h.num}</td>
                          <td style={s.td(true)}>{h.name}</td>
                          <td style={s.td(false)}>{formatPct(h.winProb)}</td>
                          <td style={s.td(false)}>{h.odds_win}倍</td>
                          <td style={s.td(true)}>
                            {formatYen(h.ev)}
                            <span style={{ fontSize: 10, color: "#6B7280", marginLeft: 4 }}>
                              ({(h.evRatio * 100).toFixed(0)}%)
                            </span>
                          </td>
                          <td style={s.td(false)}>
                            <span style={s.badge(b.color)}>{b.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Place EV */}
            <div style={s.card}>
              <div style={s.cardTitle}>複勝 期待値ランキング</div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>番</th>
                      <th style={s.th}>馬名</th>
                      <th style={s.th}>3着内確率</th>
                      <th style={s.th}>オッズ</th>
                      <th style={s.th}>期待値</th>
                      <th style={s.th}>判定</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calc.placeBets.slice(0, 8).map((h) => {
                      const b = evBadge(h.evRatio);
                      return (
                        <tr key={h.num}>
                          <td style={s.td(false)}>{h.num}</td>
                          <td style={s.td(true)}>{h.name}</td>
                          <td style={s.td(false)}>{formatPct(h.placeProb)}</td>
                          <td style={s.td(false)}>{h.odds_place}倍</td>
                          <td style={s.td(true)}>
                            {formatYen(h.ev)}
                            <span style={{ fontSize: 10, color: "#6B7280", marginLeft: 4 }}>
                              ({(h.evRatio * 100).toFixed(0)}%)
                            </span>
                          </td>
                          <td style={s.td(false)}>
                            <span style={s.badge(b.color)}>{b.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Quinella EV */}
            <div style={s.card}>
              <div style={s.cardTitle}>馬連 期待値TOP10</div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>組合せ</th>
                      <th style={s.th}>馬名</th>
                      <th style={s.th}>的中率</th>
                      <th style={s.th}>想定配当</th>
                      <th style={s.th}>期待値</th>
                      <th style={s.th}>判定</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calc.quinellas.slice(0, 10).map((q) => {
                      const b = evBadge(q.evRatio);
                      return (
                        <tr key={q.pair}>
                          <td style={s.td(true)}>{q.pair}</td>
                          <td style={{ ...s.td(false), fontSize: 11 }}>
                            {q.nameA}×{q.nameB}
                          </td>
                          <td style={s.td(false)}>{formatPct(q.prob)}</td>
                          <td style={s.td(false)}>{formatYen(q.estOdds * 100)}</td>
                          <td style={s.td(true)}>
                            {formatYen(q.ev)}
                            <span style={{ fontSize: 10, color: "#6B7280", marginLeft: 4 }}>
                              ({(q.evRatio * 100).toFixed(0)}%)
                            </span>
                          </td>
                          <td style={s.td(false)}>
                            <span style={s.badge(b.color)}>{b.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Wide EV */}
            <div style={s.card}>
              <div style={s.cardTitle}>ワイド 期待値TOP10</div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>組合せ</th>
                      <th style={s.th}>馬名</th>
                      <th style={s.th}>的中率</th>
                      <th style={s.th}>想定配当</th>
                      <th style={s.th}>期待値</th>
                      <th style={s.th}>判定</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calc.wides.slice(0, 10).map((w) => {
                      const b = evBadge(w.evRatio);
                      return (
                        <tr key={w.pair}>
                          <td style={s.td(true)}>{w.pair}</td>
                          <td style={{ ...s.td(false), fontSize: 11 }}>
                            {w.nameA}×{w.nameB}
                          </td>
                          <td style={s.td(false)}>{formatPct(w.prob)}</td>
                          <td style={s.td(false)}>{formatYen(w.estOdds * 100)}</td>
                          <td style={s.td(true)}>
                            {formatYen(w.ev)}
                            <span style={{ fontSize: 10, color: "#6B7280", marginLeft: 4 }}>
                              ({(w.evRatio * 100).toFixed(0)}%)
                            </span>
                          </td>
                          <td style={s.td(false)}>
                            <span style={s.badge(b.color)}>{b.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Trio EV */}
            <div style={s.card}>
              <div style={s.cardTitle}>3連複 期待値TOP10</div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>組合せ</th>
                      <th style={s.th}>的中率</th>
                      <th style={s.th}>想定配当</th>
                      <th style={s.th}>期待値</th>
                      <th style={s.th}>判定</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calc.trios.slice(0, 10).map((t) => {
                      const b = evBadge(t.evRatio);
                      return (
                        <tr key={t.trio}>
                          <td style={s.td(true)}>{t.trio}</td>
                          <td style={s.td(false)}>{formatPct(t.prob)}</td>
                          <td style={s.td(false)}>{formatYen(t.estOdds * 100)}</td>
                          <td style={s.td(true)}>
                            {formatYen(t.ev)}
                            <span style={{ fontSize: 10, color: "#6B7280", marginLeft: 4 }}>
                              ({(t.evRatio * 100).toFixed(0)}%)
                            </span>
                          </td>
                          <td style={s.td(false)}>
                            <span style={s.badge(b.color)}>{b.label}</span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {/* VALIDATE TAB */}
        {activeTab === "validate" && (
          <>
            <div style={s.card}>
              <div style={s.cardTitle}>小倉牝馬S2026 実結果との照合</div>
              <div style={{ fontSize: 12, color: "#9CA3AF", lineHeight: 1.8 }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12, marginBottom: 16 }}>
                  <div style={{ background: "#22C55E12", border: "1px solid #22C55E30", borderRadius: 8, padding: 12 }}>
                    <div style={{ fontSize: 11, color: "#22C55E" }}>1着 ジョスラン (17番)</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "#F9FAFB" }}>
                      {formatPct(validation.winnerProb)}
                    </div>
                    <div style={{ fontSize: 11, color: "#6B7280" }}>
                      予想順位: {validation.winnerRank}位
                    </div>
                  </div>
                  <div style={{ background: "#3B82F612", border: "1px solid #3B82F630", borderRadius: 8, padding: 12 }}>
                    <div style={{ fontSize: 11, color: "#3B82F6" }}>2着 ボンドガール (16番)</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "#F9FAFB" }}>
                      {formatPct(validation.secondProb)}
                    </div>
                    <div style={{ fontSize: 11, color: "#6B7280" }}>
                      予想順位: {validation.secondRank}位
                    </div>
                  </div>
                  <div style={{ background: "#EAB30812", border: "1px solid #EAB30830", borderRadius: 8, padding: 12 }}>
                    <div style={{ fontSize: 11, color: "#EAB308" }}>3着 ココナッツブラウン (8番)</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "#F9FAFB" }}>
                      {formatPct(validation.thirdProb)}
                    </div>
                    <div style={{ fontSize: 11, color: "#6B7280" }}>
                      予想順位: {validation.thirdRank}位
                    </div>
                  </div>
                </div>

                <div style={{ background: "#1F2937", borderRadius: 8, padding: 12, marginBottom: 12 }}>
                  <div style={{ fontSize: 13, fontWeight: 600, color: "#E5E7EB", marginBottom: 8 }}>券種別の的中確率</div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <div>馬連 17-16: <strong style={{ color: "#F9FAFB" }}>{formatPct(validation.quinellaProb)}</strong></div>
                    <div>3連複 16-17-8: <strong style={{ color: "#F9FAFB" }}>{formatPct(validation.trioProb)}</strong></div>
                    <div>ワイド 17-16: <strong style={{ color: "#F9FAFB" }}>{formatPct(validation.wideProbs[0])}</strong></div>
                    <div>ワイド 17-8: <strong style={{ color: "#F9FAFB" }}>{formatPct(validation.wideProbs[1])}</strong></div>
                    <div>ワイド 16-8: <strong style={{ color: "#F9FAFB" }}>{formatPct(validation.wideProbs[2])}</strong></div>
                  </div>
                </div>
              </div>
            </div>

            <div style={s.card}>
              <div style={s.cardTitle}>温度パラメータの最適化ヒント</div>
              <div style={{ fontSize: 12, color: "#9CA3AF", lineHeight: 1.8 }}>
                <p>現在T={temperature}で1着ジョスランの確率は{formatPct(validation.winnerProb)}です。</p>
                <p>
                  ジョスランは単勝3.6倍（市場の暗黙確率≈{formatPct(1/3.6*0.8)}）で決まったので、
                  モデルの確率がこれに近いTが適切です。
                </p>
                <p>
                  ただし2着のボンドガール（9人気）を拾えるかが本当のテスト。
                  評価点58は{validation.secondRank}位で、ここが課題です。
                </p>
                <div style={{ marginTop: 8, padding: "8px 12px", background: "#0B0E11", borderRadius: 6, border: "1px solid #1F2937" }}>
                  <div style={{ fontSize: 11, color: "#60A5FA", fontWeight: 600 }}>検証から見えること</div>
                  <div style={{ fontSize: 12, marginTop: 4, lineHeight: 1.6 }}>
                    ① 評価点が正しければ確率変換は機能する（1着・3着は予想上位）<br/>
                    ② 2着ボンドガール（評価点58）の見落としは評価点側の問題<br/>
                    ③ 温度パラメータはレースの荒れ具合を事前に判断する必要がある
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
