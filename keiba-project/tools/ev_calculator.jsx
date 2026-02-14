import { useState, useMemo, useCallback, useRef } from "react";

// ─── Empty templates matching data/templates/input.json ───
const EMPTY_RACE = {
  date: "",
  venue: "",
  race_number: 0,
  name: "",
  grade: "",
  surface: "",
  distance: 0,
  direction: "",
  entries: 0,
  weather: "",
  track_condition: "",
};

const EMPTY_HORSE = {
  num: 0,
  name: "",
  score: 50,
  score_breakdown: { ability: 25, jockey: 10, fitness: 8, form: 5, other: 2 },
  odds_win: 10.0,
  odds_place: 3.0,
  note: "",
};

// ─── Math utilities ───
function softmax(scores, temperature) {
  const maxScore = Math.max(...scores);
  const exps = scores.map((s) => Math.exp((s - maxScore) / temperature));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function calcPlaceProbs(winProbs) {
  const n = winProbs.length;
  const placeProbs = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let pTop3 = winProbs[i];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      pTop3 += winProbs[j] * (winProbs[i] / (1 - winProbs[j]));
    }
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

function calcQuinellaProb(winProbs, i, j) {
  return (
    winProbs[i] * (winProbs[j] / (1 - winProbs[i])) +
    winProbs[j] * (winProbs[i] / (1 - winProbs[j]))
  );
}

function calcWideProb(winProbs, i, j) {
  const n = winProbs.length;
  let prob = 0;
  let pJ_top2_given_I = winProbs[j] / (1 - winProbs[i]);
  for (let k = 0; k < n; k++) {
    if (k === i || k === j) continue;
    pJ_top2_given_I +=
      (winProbs[k] / (1 - winProbs[i])) *
      (winProbs[j] / (1 - winProbs[i] - winProbs[k]));
  }
  prob += winProbs[i] * pJ_top2_given_I;
  let pI_top2_given_J = winProbs[i] / (1 - winProbs[j]);
  for (let k = 0; k < n; k++) {
    if (k === i || k === j) continue;
    pI_top2_given_J +=
      (winProbs[k] / (1 - winProbs[j])) *
      (winProbs[i] / (1 - winProbs[j] - winProbs[k]));
  }
  prob += winProbs[j] * pI_top2_given_J;
  for (let k = 0; k < n; k++) {
    if (k === i || k === j) continue;
    const rest = 1 - winProbs[k];
    prob +=
      winProbs[k] *
      ((winProbs[i] / rest) * (winProbs[j] / (rest - winProbs[i])) +
        (winProbs[j] / rest) * (winProbs[i] / (rest - winProbs[j])));
  }
  return Math.min(prob, 1);
}

function calcTrioProb(winProbs, i, j, k) {
  const perms = [
    [i, j, k],
    [i, k, j],
    [j, i, k],
    [j, k, i],
    [k, i, j],
    [k, j, i],
  ];
  let prob = 0;
  for (const [a, b, c] of perms) {
    prob +=
      winProbs[a] *
      (winProbs[b] / (1 - winProbs[a])) *
      (winProbs[c] / (1 - winProbs[a] - winProbs[b]));
  }
  return prob;
}

function formatPct(v) {
  return (v * 100).toFixed(1) + "%";
}
function formatYen(v) {
  return "¥" + Math.round(v).toLocaleString();
}

// ─── Grade presets (auto-set budget & temperature) ───
const GRADE_PRESETS = [
  { label: "G1", budget: 10000, temp: 8 },
  { label: "G2", budget: 5000, temp: 8 },
  { label: "G3", budget: 3000, temp: 10 },
  { label: "OP/L", budget: 1500, temp: 10 },
  { label: "条件戦", budget: 1500, temp: 12 },
  { label: "未勝利", budget: 1500, temp: 12 },
];

const TABS = [
  { id: "race", label: "レース情報" },
  { id: "input", label: "評価点入力" },
  { id: "prob", label: "確率変換" },
  { id: "ev", label: "期待値計算" },
  { id: "validate", label: "検証" },
];

// ─── Main Component ───
export default function EVCalculator() {
  const [race, setRace] = useState({ ...EMPTY_RACE });
  const [horses, setHorses] = useState([]);
  const [temperature, setTemperature] = useState(10);
  const [budget, setBudget] = useState(3000);
  const [activeTab, setActiveTab] = useState("race");
  const [topN, setTopN] = useState(6);
  const [betUnit] = useState(100);
  const [actualResult, setActualResult] = useState({
    first: 0,
    second: 0,
    third: 0,
  });
  const [jsonInput, setJsonInput] = useState("");
  const [importError, setImportError] = useState("");
  const [editingBreakdown, setEditingBreakdown] = useState(null);
  const fileRef = useRef(null);

  // ─── Handlers ───
  const updateRace = useCallback((field, value) => {
    setRace((prev) => ({ ...prev, [field]: value }));
  }, []);

  const updateHorse = useCallback((idx, field, value) => {
    setHorses((prev) => {
      const next = [...prev];
      next[idx] = {
        ...next[idx],
        [field]:
          field === "name" || field === "note" ? value : Number(value) || 0,
      };
      return next;
    });
  }, []);

  const updateBreakdown = useCallback((idx, key, value) => {
    setHorses((prev) => {
      const next = [...prev];
      const bd = { ...next[idx].score_breakdown, [key]: Number(value) || 0 };
      const total = bd.ability + bd.jockey + bd.fitness + bd.form + bd.other;
      next[idx] = { ...next[idx], score_breakdown: bd, score: total };
      return next;
    });
  }, []);

  const addHorse = useCallback(() => {
    setHorses((prev) => [
      ...prev,
      {
        ...EMPTY_HORSE,
        num: prev.length + 1,
        score_breakdown: { ...EMPTY_HORSE.score_breakdown },
      },
    ]);
  }, []);

  const removeHorse = useCallback((idx) => {
    setHorses((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const importJSON = useCallback((jsonStr) => {
    try {
      setImportError("");
      const data = JSON.parse(jsonStr);
      if (data.race) setRace({ ...EMPTY_RACE, ...data.race });
      if (data.parameters) {
        if (data.parameters.temperature)
          setTemperature(data.parameters.temperature);
        if (data.parameters.budget) setBudget(data.parameters.budget);
        if (data.parameters.top_n) setTopN(data.parameters.top_n);
      }
      if (data.horses && Array.isArray(data.horses)) {
        setHorses(
          data.horses.map((h) => ({
            ...EMPTY_HORSE,
            ...h,
            score_breakdown: {
              ...EMPTY_HORSE.score_breakdown,
              ...(h.score_breakdown || {}),
            },
          }))
        );
        setActiveTab("input");
      }
    } catch (e) {
      setImportError("JSONパースエラー: " + e.message);
    }
  }, []);

  const handleFileImport = useCallback(
    (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        const text = ev.target.result;
        setJsonInput(text);
        importJSON(text);
      };
      reader.readAsText(file);
      e.target.value = "";
    },
    [importJSON]
  );

  const exportJSON = useCallback(() => {
    const data = {
      race,
      parameters: { temperature, budget, top_n: topN },
      horses: horses.map(
        ({ num, name, score, score_breakdown, odds_win, odds_place, note }) => ({
          num,
          name,
          score,
          score_breakdown,
          odds_win,
          odds_place,
          note,
        })
      ),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const fname =
      race.date && race.venue && race.race_number
        ? `${race.date.replace(/-/g, "")}_${race.venue}_${race.race_number}R_input.json`
        : "input.json";
    a.href = url;
    a.download = fname;
    a.click();
    URL.revokeObjectURL(url);
  }, [race, horses, temperature, budget, topN]);

  // ─── Core calculations ───
  const calc = useMemo(() => {
    if (horses.length < 2) return null;
    const scores = horses.map((h) => h.score);
    const winProbs = softmax(scores, temperature);
    const placeProbs = calcPlaceProbs(winProbs);

    const ranked = horses
      .map((h, i) => ({
        ...h,
        idx: i,
        winProb: winProbs[i],
        placeProb: placeProbs[i],
      }))
      .sort((a, b) => b.score - a.score);

    const topHorses = ranked.slice(0, Math.min(topN, ranked.length));
    const topIndices = topHorses.map((h) => h.idx);

    // Market probability comparison
    const marketComparison = ranked.map((h) => {
      const marketProb = h.odds_win > 0 ? (1 / h.odds_win) * 0.8 : 0;
      const diff = h.winProb - marketProb;
      return {
        ...h,
        marketProb,
        diff,
        flag: diff > 0.05 ? "💎" : diff < -0.05 ? "⚠️" : "",
      };
    });

    // Quinella combinations
    const quinellas = [];
    for (let a = 0; a < topIndices.length; a++) {
      for (let b = a + 1; b < topIndices.length; b++) {
        const i = topIndices[a],
          j = topIndices[b];
        const prob = calcQuinellaProb(winProbs, i, j);
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
        const i = topIndices[a],
          j = topIndices[b];
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

    // Trio combinations
    const trioIndices = topIndices.slice(0, Math.min(topN, 7));
    const trios = [];
    for (let a = 0; a < trioIndices.length; a++) {
      for (let b = a + 1; b < trioIndices.length; b++) {
        for (let c = b + 1; c < trioIndices.length; c++) {
          const i = trioIndices[a],
            j = trioIndices[b],
            k = trioIndices[c];
          const prob = calcTrioProb(winProbs, i, j, k);
          const estOdds = Math.max(
            (horses[i].odds_win *
              horses[j].odds_win *
              horses[k].odds_win) /
              200,
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

    // Win & Place EV
    const winBets = ranked.map((h) => {
      const ev = betUnit * h.odds_win * h.winProb;
      return { ...h, ev, evRatio: ev / betUnit };
    });
    const placeBets = ranked.map((h) => {
      const ev = betUnit * h.odds_place * h.placeProb;
      return { ...h, ev, evRatio: ev / betUnit };
    });

    // Confidence
    const entropy = -winProbs.reduce(
      (s, p) => s + (p > 0 ? p * Math.log2(p) : 0),
      0
    );
    const maxEntropy = Math.log2(horses.length);
    const confidence = 1 - entropy / maxEntropy;

    return {
      winProbs,
      placeProbs,
      ranked,
      marketComparison,
      quinellas,
      wides,
      trios,
      winBets,
      placeBets,
      confidence,
    };
  }, [horses, temperature, topN, betUnit]);

  // ─── Validation ───
  const validation = useMemo(() => {
    if (!calc || !actualResult.first) return null;
    const { first, second, third } = actualResult;
    const wI = horses.findIndex((h) => h.num === first);
    const sI = horses.findIndex((h) => h.num === second);
    const tI = horses.findIndex((h) => h.num === third);
    if (wI < 0 || sI < 0 || tI < 0) return null;

    const top3Nums = calc.ranked.slice(0, 3).map((h) => h.num);
    const actualNums = [first, second, third];
    const top3InFrame = actualNums.every((n) => top3Nums.includes(n));

    return {
      winnerName: horses[wI].name,
      winnerNum: first,
      winnerProb: calc.winProbs[wI],
      winnerRank: calc.ranked.findIndex((h) => h.num === first) + 1,
      secondName: horses[sI].name,
      secondNum: second,
      secondProb: calc.winProbs[sI],
      secondRank: calc.ranked.findIndex((h) => h.num === second) + 1,
      thirdName: horses[tI].name,
      thirdNum: third,
      thirdProb: calc.winProbs[tI],
      thirdRank: calc.ranked.findIndex((h) => h.num === third) + 1,
      quinellaProb: calcQuinellaProb(calc.winProbs, wI, sI),
      wideProbs: [
        calcWideProb(calc.winProbs, wI, sI),
        calcWideProb(calc.winProbs, wI, tI),
        calcWideProb(calc.winProbs, sI, tI),
      ],
      trioProb: calcTrioProb(calc.winProbs, wI, sI, tI),
      top3InFrame,
    };
  }, [calc, actualResult, horses]);

  // ─── Styles ───
  const s = {
    root: {
      fontFamily: "'Noto Sans JP', 'Hiragino Sans', system-ui, sans-serif",
      background: "#0B0E11",
      color: "#D1D5DB",
      minHeight: "100vh",
      padding: "16px",
    },
    header: {
      maxWidth: 1000,
      margin: "0 auto",
      paddingBottom: 14,
      borderBottom: "1px solid #1F2937",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "flex-end",
      flexWrap: "wrap",
      gap: 8,
    },
    title: { fontSize: 20, fontWeight: 700, color: "#F9FAFB", margin: 0 },
    subtitle: { fontSize: 12, color: "#6B7280", marginTop: 4 },
    main: { maxWidth: 1000, margin: "0 auto", marginTop: 16 },
    tabs: {
      display: "flex",
      gap: 2,
      marginBottom: 16,
      borderBottom: "1px solid #1F2937",
      overflowX: "auto",
    },
    tab: (active) => ({
      padding: "8px 14px",
      fontSize: 13,
      fontWeight: active ? 600 : 400,
      color: active ? "#60A5FA" : "#6B7280",
      background: "transparent",
      border: "none",
      borderBottom: active
        ? "2px solid #60A5FA"
        : "2px solid transparent",
      cursor: "pointer",
      whiteSpace: "nowrap",
    }),
    card: {
      background: "#111827",
      border: "1px solid #1F2937",
      borderRadius: 10,
      padding: 16,
      marginBottom: 12,
    },
    cardTitle: {
      fontSize: 14,
      fontWeight: 600,
      color: "#E5E7EB",
      marginBottom: 10,
    },
    table: { width: "100%", borderCollapse: "collapse", fontSize: 12 },
    th: {
      textAlign: "left",
      padding: "6px 8px",
      borderBottom: "1px solid #1F2937",
      color: "#9CA3AF",
      fontWeight: 500,
      fontSize: 11,
      whiteSpace: "nowrap",
    },
    td: (hl) => ({
      padding: "5px 8px",
      borderBottom: "1px solid #111827",
      color: hl ? "#F9FAFB" : "#D1D5DB",
      fontWeight: hl ? 600 : 400,
    }),
    input: {
      width: 54,
      padding: "3px 6px",
      fontSize: 12,
      background: "#1F2937",
      border: "1px solid #374151",
      borderRadius: 4,
      color: "#F9FAFB",
      textAlign: "right",
    },
    inputFull: {
      width: "100%",
      padding: "4px 8px",
      fontSize: 12,
      background: "#1F2937",
      border: "1px solid #374151",
      borderRadius: 4,
      color: "#F9FAFB",
      boxSizing: "border-box",
    },
    textarea: {
      width: "100%",
      minHeight: 120,
      padding: "8px",
      fontSize: 11,
      fontFamily: "monospace",
      background: "#1F2937",
      border: "1px solid #374151",
      borderRadius: 6,
      color: "#F9FAFB",
      resize: "vertical",
      boxSizing: "border-box",
    },
    btn: (color = "#3B82F6") => ({
      padding: "6px 14px",
      fontSize: 12,
      fontWeight: 600,
      background: color + "20",
      color,
      border: `1px solid ${color}50`,
      borderRadius: 6,
      cursor: "pointer",
    }),
    btnSmall: (color = "#6B7280") => ({
      padding: "2px 8px",
      fontSize: 10,
      background: color + "15",
      color,
      border: `1px solid ${color}30`,
      borderRadius: 4,
      cursor: "pointer",
    }),
    badge: (color) => ({
      display: "inline-block",
      padding: "1px 6px",
      borderRadius: 3,
      fontSize: 10,
      fontWeight: 600,
      color,
      background: color + "18",
      border: `1px solid ${color}40`,
    }),
    slider: { width: "100%", accentColor: "#3B82F6" },
    label: { fontSize: 11, color: "#9CA3AF", marginBottom: 4, display: "block" },
    fieldRow: {
      display: "grid",
      gridTemplateColumns: "1fr 1fr 1fr 1fr",
      gap: 10,
      marginBottom: 10,
    },
  };

  const evBadge = (ratio) => {
    if (ratio >= 1.5) return { label: "S級", color: "#22C55E" };
    if (ratio >= 1.2) return { label: "A級", color: "#3B82F6" };
    if (ratio >= 1.0) return { label: "B級", color: "#EAB308" };
    return { label: "C級", color: "#6B7280" };
  };

  const raceName =
    race.name ||
    (race.venue && race.race_number
      ? `${race.venue}${race.race_number}R`
      : "未設定");

  return (
    <div style={s.root}>
      {/* ═══ Header ═══ */}
      <div style={s.header}>
        <div>
          <h1 style={s.title}>馬券期待値計算機 v1.0</h1>
          <p style={s.subtitle}>
            {horses.length > 0
              ? `${raceName}　${horses.length}頭`
              : "レースデータをインポートまたは入力してください"}
          </p>
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <input
            type="file"
            ref={fileRef}
            accept=".json"
            onChange={handleFileImport}
            style={{ display: "none" }}
          />
          <button
            style={s.btn("#8B5CF6")}
            onClick={() => fileRef.current?.click()}
          >
            JSONインポート
          </button>
          {horses.length > 0 && (
            <button style={s.btn("#10B981")} onClick={exportJSON}>
              JSONエクスポート
            </button>
          )}
        </div>
      </div>

      <div style={s.main}>
        {/* ═══ Tabs ═══ */}
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

        {/* ═══ Controls bar ═══ */}
        {horses.length >= 2 && (
          <div
            style={{
              ...s.card,
              display: "flex",
              gap: 20,
              alignItems: "center",
              flexWrap: "wrap",
            }}
          >
            <div>
              <div style={s.label}>温度 T={temperature}</div>
              <input
                type="range"
                min="2"
                max="25"
                step="0.5"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                style={{ ...s.slider, width: 140 }}
              />
              <div style={{ fontSize: 10, color: "#6B7280" }}>
                低=実力差拡大 高=混戦
              </div>
            </div>
            <div>
              <div style={s.label}>上位N頭</div>
              <input
                type="range"
                min="3"
                max={Math.min(horses.length, 10)}
                step="1"
                value={topN}
                onChange={(e) => setTopN(Number(e.target.value))}
                style={{ ...s.slider, width: 100 }}
              />
              <span style={{ fontSize: 12, color: "#E5E7EB", marginLeft: 6 }}>
                {topN}頭
              </span>
            </div>
            <div>
              <div style={s.label}>予算</div>
              <input
                type="number"
                value={budget}
                step={500}
                onChange={(e) => setBudget(Number(e.target.value))}
                style={{ ...s.input, width: 70 }}
              />
              <span style={{ fontSize: 11, color: "#6B7280", marginLeft: 4 }}>
                円
              </span>
            </div>
            {calc && (
              <div style={{ marginLeft: "auto", textAlign: "right" }}>
                <div style={{ fontSize: 11, color: "#9CA3AF" }}>
                  レース確信度
                </div>
                <div
                  style={{
                    fontSize: 22,
                    fontWeight: 700,
                    color:
                      calc.confidence > 0.3
                        ? "#22C55E"
                        : calc.confidence > 0.15
                          ? "#EAB308"
                          : "#EF4444",
                  }}
                >
                  {(calc.confidence * 100).toFixed(0)}%
                </div>
                <div style={{ fontSize: 10, color: "#6B7280" }}>
                  {calc.confidence > 0.3
                    ? "高確信"
                    : calc.confidence > 0.15
                      ? "中確信"
                      : "低確信・見送り検討"}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══════════════ RACE INFO TAB ═══════════════ */}
        {activeTab === "race" && (
          <>
            <div style={s.card}>
              <div style={s.cardTitle}>レース情報</div>
              <div style={s.fieldRow}>
                <div>
                  <div style={s.label}>日付</div>
                  <input
                    type="date"
                    value={race.date}
                    onChange={(e) => updateRace("date", e.target.value)}
                    style={s.inputFull}
                  />
                </div>
                <div>
                  <div style={s.label}>競馬場</div>
                  <input
                    value={race.venue}
                    onChange={(e) => updateRace("venue", e.target.value)}
                    placeholder="京都"
                    style={s.inputFull}
                  />
                </div>
                <div>
                  <div style={s.label}>レース番号</div>
                  <input
                    type="number"
                    value={race.race_number || ""}
                    onChange={(e) =>
                      updateRace("race_number", Number(e.target.value))
                    }
                    style={s.inputFull}
                  />
                </div>
                <div>
                  <div style={s.label}>レース名</div>
                  <input
                    value={race.name}
                    onChange={(e) => updateRace("name", e.target.value)}
                    placeholder="3歳未勝利"
                    style={s.inputFull}
                  />
                </div>
              </div>
              <div style={s.fieldRow}>
                <div>
                  <div style={s.label}>グレード</div>
                  <select
                    value={race.grade}
                    onChange={(e) => {
                      updateRace("grade", e.target.value);
                      const preset = GRADE_PRESETS.find(
                        (p) => p.label === e.target.value
                      );
                      if (preset) {
                        setBudget(preset.budget);
                        setTemperature(preset.temp);
                      }
                    }}
                    style={s.inputFull}
                  >
                    <option value="">選択</option>
                    {GRADE_PRESETS.map((g) => (
                      <option key={g.label} value={g.label}>
                        {g.label}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <div style={s.label}>馬場</div>
                  <select
                    value={race.surface}
                    onChange={(e) => updateRace("surface", e.target.value)}
                    style={s.inputFull}
                  >
                    <option value="">選択</option>
                    <option value="芝">芝</option>
                    <option value="ダート">ダート</option>
                  </select>
                </div>
                <div>
                  <div style={s.label}>距離 (m)</div>
                  <input
                    type="number"
                    value={race.distance || ""}
                    onChange={(e) =>
                      updateRace("distance", Number(e.target.value))
                    }
                    style={s.inputFull}
                  />
                </div>
                <div>
                  <div style={s.label}>回り</div>
                  <select
                    value={race.direction}
                    onChange={(e) => updateRace("direction", e.target.value)}
                    style={s.inputFull}
                  >
                    <option value="">選択</option>
                    <option value="右">右</option>
                    <option value="左">左</option>
                    <option value="直線">直線</option>
                  </select>
                </div>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: 10,
                }}
              >
                <div>
                  <div style={s.label}>天候</div>
                  <input
                    value={race.weather}
                    onChange={(e) => updateRace("weather", e.target.value)}
                    placeholder="晴"
                    style={s.inputFull}
                  />
                </div>
                <div>
                  <div style={s.label}>馬場状態</div>
                  <input
                    value={race.track_condition}
                    onChange={(e) =>
                      updateRace("track_condition", e.target.value)
                    }
                    placeholder="良"
                    style={s.inputFull}
                  />
                </div>
              </div>
            </div>

            {/* JSON Import */}
            <div style={s.card}>
              <div style={s.cardTitle}>JSONデータ入力</div>
              <p style={{ fontSize: 11, color: "#6B7280", marginBottom: 8 }}>
                input.jsonフォーマットのデータを貼り付けてインポートできます。
                上部の「JSONインポート」ボタンからファイル読み込みも可能です。
              </p>
              <textarea
                style={s.textarea}
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                placeholder='{"race": {...}, "parameters": {...}, "horses": [...]}'
              />
              {importError && (
                <div style={{ color: "#EF4444", fontSize: 11, marginTop: 6 }}>
                  {importError}
                </div>
              )}
              <div style={{ marginTop: 8, display: "flex", gap: 8 }}>
                <button
                  style={s.btn("#3B82F6")}
                  onClick={() => importJSON(jsonInput)}
                >
                  インポート実行
                </button>
                <button
                  style={s.btn("#6B7280")}
                  onClick={() => {
                    setJsonInput("");
                    setImportError("");
                  }}
                >
                  クリア
                </button>
              </div>
            </div>

            {horses.length === 0 && (
              <div style={s.card}>
                <div style={s.cardTitle}>出走馬の追加</div>
                <p style={{ fontSize: 12, color: "#9CA3AF", marginBottom: 10 }}>
                  JSONインポートするか、手動で馬を追加してください。
                </p>
                <button style={s.btn("#3B82F6")} onClick={addHorse}>
                  最初の馬を追加
                </button>
              </div>
            )}
          </>
        )}

        {/* ═══════════════ INPUT TAB ═══════════════ */}
        {activeTab === "input" && (
          <div style={s.card}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 10,
              }}
            >
              <div style={s.cardTitle}>
                評価点・オッズ入力（{horses.length}頭）
              </div>
              <button style={s.btn("#3B82F6")} onClick={addHorse}>
                ＋ 馬を追加
              </button>
            </div>
            {horses.length === 0 ? (
              <p style={{ fontSize: 12, color: "#6B7280" }}>
                「レース情報」タブからデータをインポートするか、「＋
                馬を追加」ボタンで入力を開始してください。
              </p>
            ) : (
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>番</th>
                      <th style={s.th}>馬名</th>
                      <th style={{ ...s.th, textAlign: "center" }}>評価点</th>
                      <th style={s.th}>内訳</th>
                      <th style={s.th}>単勝</th>
                      <th style={s.th}>複勝</th>
                      <th style={s.th}>メモ</th>
                      <th style={s.th}></th>
                    </tr>
                  </thead>
                  <tbody>
                    {horses.map((h, i) => (
                      <tr key={i}>
                        <td style={s.td(false)}>
                          <input
                            style={{ ...s.input, width: 36 }}
                            value={h.num}
                            onChange={(e) =>
                              updateHorse(i, "num", e.target.value)
                            }
                          />
                        </td>
                        <td style={s.td(true)}>
                          <input
                            style={{
                              ...s.input,
                              width: 100,
                              textAlign: "left",
                            }}
                            value={h.name}
                            onChange={(e) =>
                              updateHorse(i, "name", e.target.value)
                            }
                          />
                        </td>
                        <td style={{ ...s.td(true), textAlign: "center" }}>
                          <span
                            style={{
                              fontSize: 16,
                              fontWeight: 700,
                              color:
                                h.score >= 75
                                  ? "#22C55E"
                                  : h.score >= 60
                                    ? "#3B82F6"
                                    : h.score >= 45
                                      ? "#EAB308"
                                      : "#6B7280",
                            }}
                          >
                            {h.score}
                          </span>
                        </td>
                        <td style={s.td(false)}>
                          <button
                            style={s.btnSmall(
                              editingBreakdown === i ? "#60A5FA" : "#6B7280"
                            )}
                            onClick={() =>
                              setEditingBreakdown(
                                editingBreakdown === i ? null : i
                              )
                            }
                          >
                            {h.score_breakdown.ability}/
                            {h.score_breakdown.jockey}/
                            {h.score_breakdown.fitness}/
                            {h.score_breakdown.form}/{h.score_breakdown.other}
                          </button>
                        </td>
                        <td style={s.td(false)}>
                          <input
                            style={s.input}
                            value={h.odds_win}
                            onChange={(e) =>
                              updateHorse(i, "odds_win", e.target.value)
                            }
                          />
                        </td>
                        <td style={s.td(false)}>
                          <input
                            style={s.input}
                            value={h.odds_place}
                            onChange={(e) =>
                              updateHorse(i, "odds_place", e.target.value)
                            }
                          />
                        </td>
                        <td style={s.td(false)}>
                          <input
                            style={{
                              ...s.input,
                              width: 140,
                              textAlign: "left",
                              fontSize: 10,
                            }}
                            value={h.note}
                            onChange={(e) =>
                              updateHorse(i, "note", e.target.value)
                            }
                          />
                        </td>
                        <td style={s.td(false)}>
                          <button
                            style={s.btnSmall("#EF4444")}
                            onClick={() => removeHorse(i)}
                          >
                            ✕
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Breakdown editor */}
            {editingBreakdown !== null && horses[editingBreakdown] && (
              <div
                style={{
                  marginTop: 12,
                  padding: 12,
                  background: "#0B0E11",
                  borderRadius: 8,
                  border: "1px solid #374151",
                }}
              >
                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#E5E7EB",
                    marginBottom: 8,
                  }}
                >
                  {horses[editingBreakdown].name ||
                    `馬番${horses[editingBreakdown].num}`}{" "}
                  — 評価内訳（合計: {horses[editingBreakdown].score}点）
                </div>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(5, 1fr)",
                    gap: 8,
                  }}
                >
                  {[
                    { key: "ability", label: "実力", max: 50 },
                    { key: "jockey", label: "騎手", max: 20 },
                    { key: "fitness", label: "適性", max: 15 },
                    { key: "form", label: "調子", max: 10 },
                    { key: "other", label: "他", max: 5 },
                  ].map((f) => (
                    <div key={f.key}>
                      <div
                        style={{
                          fontSize: 10,
                          color: "#9CA3AF",
                          marginBottom: 2,
                        }}
                      >
                        {f.label} (/{f.max})
                      </div>
                      <input
                        type="number"
                        min="0"
                        max={f.max}
                        style={{ ...s.input, width: "100%" }}
                        value={
                          horses[editingBreakdown].score_breakdown[f.key]
                        }
                        onChange={(e) =>
                          updateBreakdown(
                            editingBreakdown,
                            f.key,
                            e.target.value
                          )
                        }
                      />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ═══════════════ PROBABILITY TAB ═══════════════ */}
        {activeTab === "prob" && calc && (
          <>
            <div style={s.card}>
              <div style={s.cardTitle}>
                確率変換結果（ソフトマックス T={temperature}）
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>順位</th>
                      <th style={s.th}>番</th>
                      <th style={s.th}>馬名</th>
                      <th style={s.th}>点数</th>
                      <th style={s.th}>1着確率</th>
                      <th style={s.th}>市場確率</th>
                      <th style={s.th}>乖離</th>
                      <th style={s.th}>3着内</th>
                      <th style={s.th}>確率バー</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calc.marketComparison.map((h, i) => (
                      <tr
                        key={h.num}
                        style={{
                          background:
                            actualResult.first === h.num
                              ? "#22C55E12"
                              : actualResult.second === h.num
                                ? "#3B82F612"
                                : actualResult.third === h.num
                                  ? "#EAB30812"
                                  : "transparent",
                        }}
                      >
                        <td style={s.td(false)}>{i + 1}</td>
                        <td style={s.td(false)}>{h.num}</td>
                        <td style={s.td(true)}>{h.name}</td>
                        <td style={s.td(false)}>{h.score}</td>
                        <td style={s.td(true)}>{formatPct(h.winProb)}</td>
                        <td style={s.td(false)}>{formatPct(h.marketProb)}</td>
                        <td style={s.td(false)}>
                          {h.flag && (
                            <span style={{ marginRight: 2 }}>{h.flag}</span>
                          )}
                          <span
                            style={{
                              color:
                                h.diff > 0
                                  ? "#22C55E"
                                  : h.diff < 0
                                    ? "#EF4444"
                                    : "#6B7280",
                            }}
                          >
                            {h.diff > 0 ? "+" : ""}
                            {(h.diff * 100).toFixed(1)}%
                          </span>
                        </td>
                        <td style={s.td(true)}>{formatPct(h.placeProb)}</td>
                        <td style={{ ...s.td(false), width: 100 }}>
                          <div
                            style={{
                              background: "#1F2937",
                              borderRadius: 2,
                              height: 6,
                            }}
                          >
                            <div
                              style={{
                                height: 6,
                                borderRadius: 2,
                                background:
                                  "linear-gradient(90deg, #3B82F6, #60A5FA)",
                                width: `${Math.min((h.winProb / Math.max(...calc.winProbs)) * 100, 100)}%`,
                              }}
                            />
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {actualResult.first > 0 && (
                <div
                  style={{ fontSize: 11, color: "#6B7280", marginTop: 8 }}
                >
                  背景色: 🟢1着 🔵2着 🟡3着（実結果）
                </div>
              )}
            </div>

            <div style={s.card}>
              <div style={s.cardTitle}>
                💎 過小評価（妙味あり）/ ⚠️ 過大評価
              </div>
              <div style={{ fontSize: 12, color: "#9CA3AF", lineHeight: 1.7 }}>
                {calc.marketComparison.filter((h) => h.flag).length === 0 ? (
                  <span>
                    モデル確率と市場確率の乖離が±5%以上の馬はいません。
                  </span>
                ) : (
                  calc.marketComparison
                    .filter((h) => h.flag)
                    .map((h) => (
                      <div key={h.num} style={{ marginBottom: 4 }}>
                        {h.flag}{" "}
                        <strong style={{ color: "#F9FAFB" }}>
                          {h.num}番 {h.name}
                        </strong>
                        　モデル {formatPct(h.winProb)} vs 市場{" "}
                        {formatPct(h.marketProb)}
                        （{h.diff > 0 ? "過小評価" : "過大評価"}）
                      </div>
                    ))
                )}
              </div>
            </div>
          </>
        )}
        {activeTab === "prob" && !calc && (
          <div style={s.card}>
            <p style={{ fontSize: 12, color: "#6B7280" }}>
              2頭以上のデータを入力してください。
            </p>
          </div>
        )}

        {/* ═══════════════ EV TAB ═══════════════ */}
        {activeTab === "ev" && calc && (
          <>
            {/* Win */}
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
                            {formatYen(h.ev)}{" "}
                            <span
                              style={{ fontSize: 10, color: "#6B7280" }}
                            >
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
            {/* Place */}
            <div style={s.card}>
              <div style={s.cardTitle}>複勝 期待値ランキング</div>
              <div style={{ overflowX: "auto" }}>
                <table style={s.table}>
                  <thead>
                    <tr>
                      <th style={s.th}>番</th>
                      <th style={s.th}>馬名</th>
                      <th style={s.th}>3着内</th>
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
                          <td style={s.td(false)}>
                            {formatPct(h.placeProb)}
                          </td>
                          <td style={s.td(false)}>{h.odds_place}倍</td>
                          <td style={s.td(true)}>
                            {formatYen(h.ev)}{" "}
                            <span
                              style={{ fontSize: 10, color: "#6B7280" }}
                            >
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
            {/* Quinella */}
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
                          <td style={s.td(false)}>
                            {formatYen(q.estOdds * 100)}
                          </td>
                          <td style={s.td(true)}>
                            {formatYen(q.ev)}{" "}
                            <span
                              style={{ fontSize: 10, color: "#6B7280" }}
                            >
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
            {/* Wide */}
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
                          <td style={s.td(false)}>
                            {formatYen(w.estOdds * 100)}
                          </td>
                          <td style={s.td(true)}>
                            {formatYen(w.ev)}{" "}
                            <span
                              style={{ fontSize: 10, color: "#6B7280" }}
                            >
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
            {/* Trio */}
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
                          <td style={s.td(false)}>
                            {formatYen(t.estOdds * 100)}
                          </td>
                          <td style={s.td(true)}>
                            {formatYen(t.ev)}{" "}
                            <span
                              style={{ fontSize: 10, color: "#6B7280" }}
                            >
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
        {activeTab === "ev" && !calc && (
          <div style={s.card}>
            <p style={{ fontSize: 12, color: "#6B7280" }}>
              2頭以上のデータを入力してください。
            </p>
          </div>
        )}

        {/* ═══════════════ VALIDATE TAB ═══════════════ */}
        {activeTab === "validate" && (
          <>
            <div style={s.card}>
              <div style={s.cardTitle}>実結果の入力</div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr 1fr",
                  gap: 12,
                }}
              >
                {[
                  { key: "first", label: "1着 馬番", color: "#22C55E" },
                  { key: "second", label: "2着 馬番", color: "#3B82F6" },
                  { key: "third", label: "3着 馬番", color: "#EAB308" },
                ].map((f) => (
                  <div key={f.key}>
                    <div
                      style={{
                        fontSize: 11,
                        color: f.color,
                        marginBottom: 4,
                      }}
                    >
                      {f.label}
                    </div>
                    <select
                      value={actualResult[f.key]}
                      onChange={(e) =>
                        setActualResult((prev) => ({
                          ...prev,
                          [f.key]: Number(e.target.value),
                        }))
                      }
                      style={{ ...s.inputFull, padding: "6px 8px" }}
                    >
                      <option value={0}>— 選択 —</option>
                      {horses.map((h) => (
                        <option key={h.num} value={h.num}>
                          {h.num} {h.name}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {validation && calc && (
              <>
                <div style={s.card}>
                  <div style={s.cardTitle}>
                    {raceName} — 実結果との照合
                  </div>
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr 1fr",
                      gap: 12,
                      marginBottom: 16,
                    }}
                  >
                    {[
                      {
                        label: `1着 ${validation.winnerName} (${validation.winnerNum}番)`,
                        prob: validation.winnerProb,
                        rank: validation.winnerRank,
                        color: "#22C55E",
                      },
                      {
                        label: `2着 ${validation.secondName} (${validation.secondNum}番)`,
                        prob: validation.secondProb,
                        rank: validation.secondRank,
                        color: "#3B82F6",
                      },
                      {
                        label: `3着 ${validation.thirdName} (${validation.thirdNum}番)`,
                        prob: validation.thirdProb,
                        rank: validation.thirdRank,
                        color: "#EAB308",
                      },
                    ].map((r, idx) => (
                      <div
                        key={idx}
                        style={{
                          background: r.color + "12",
                          border: `1px solid ${r.color}30`,
                          borderRadius: 8,
                          padding: 12,
                        }}
                      >
                        <div style={{ fontSize: 11, color: r.color }}>
                          {r.label}
                        </div>
                        <div
                          style={{
                            fontSize: 18,
                            fontWeight: 700,
                            color: "#F9FAFB",
                          }}
                        >
                          {formatPct(r.prob)}
                        </div>
                        <div style={{ fontSize: 11, color: "#6B7280" }}>
                          予想順位: {r.rank}位
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Top 3 check */}
                  <div
                    style={{
                      background: validation.top3InFrame
                        ? "#22C55E12"
                        : "#EF444412",
                      border: `1px solid ${validation.top3InFrame ? "#22C55E30" : "#EF444430"}`,
                      borderRadius: 8,
                      padding: 12,
                      marginBottom: 12,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: validation.top3InFrame
                          ? "#22C55E"
                          : "#EF4444",
                      }}
                    >
                      {validation.top3InFrame
                        ? "✅ 評価点上位3頭が3着内を独占"
                        : "❌ 評価点上位3頭は3着内を独占できず"}
                    </div>
                    <div
                      style={{
                        fontSize: 11,
                        color: "#9CA3AF",
                        marginTop: 4,
                      }}
                    >
                      評価上位3頭:{" "}
                      {calc.ranked
                        .slice(0, 3)
                        .map((h) => `${h.num}番${h.name}`)
                        .join("、")}
                    </div>
                  </div>

                  {/* Ticket probabilities */}
                  <div
                    style={{
                      background: "#1F2937",
                      borderRadius: 8,
                      padding: 12,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 13,
                        fontWeight: 600,
                        color: "#E5E7EB",
                        marginBottom: 8,
                      }}
                    >
                      券種別の的中確率
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "1fr 1fr",
                        gap: 8,
                        fontSize: 12,
                        color: "#9CA3AF",
                      }}
                    >
                      <div>
                        馬連 {validation.winnerNum}-{validation.secondNum}:{" "}
                        <strong style={{ color: "#F9FAFB" }}>
                          {formatPct(validation.quinellaProb)}
                        </strong>
                      </div>
                      <div>
                        3連複 {validation.winnerNum}-{validation.secondNum}-
                        {validation.thirdNum}:{" "}
                        <strong style={{ color: "#F9FAFB" }}>
                          {formatPct(validation.trioProb)}
                        </strong>
                      </div>
                      <div>
                        ワイド {validation.winnerNum}-{validation.secondNum}:{" "}
                        <strong style={{ color: "#F9FAFB" }}>
                          {formatPct(validation.wideProbs[0])}
                        </strong>
                      </div>
                      <div>
                        ワイド {validation.winnerNum}-{validation.thirdNum}:{" "}
                        <strong style={{ color: "#F9FAFB" }}>
                          {formatPct(validation.wideProbs[1])}
                        </strong>
                      </div>
                      <div>
                        ワイド {validation.secondNum}-{validation.thirdNum}:{" "}
                        <strong style={{ color: "#F9FAFB" }}>
                          {formatPct(validation.wideProbs[2])}
                        </strong>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Validation summary */}
                <div style={s.card}>
                  <div style={s.cardTitle}>検証サマリー</div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "#9CA3AF",
                      lineHeight: 1.8,
                    }}
                  >
                    <p>
                      T={temperature} での1着{validation.winnerName}の確率は{" "}
                      {formatPct(validation.winnerProb)}（予想
                      {validation.winnerRank}位）。
                    </p>
                    <div
                      style={{
                        marginTop: 8,
                        padding: "8px 12px",
                        background: "#0B0E11",
                        borderRadius: 6,
                        border: "1px solid #1F2937",
                      }}
                    >
                      <div
                        style={{
                          fontSize: 11,
                          color: "#60A5FA",
                          fontWeight: 600,
                        }}
                      >
                        チェックポイント
                      </div>
                      <div
                        style={{
                          fontSize: 12,
                          marginTop: 4,
                          lineHeight: 1.6,
                        }}
                      >
                        ① 評価点上位3頭が3着内 →{" "}
                        {validation.top3InFrame ? "はい ✅" : "いいえ ❌"}
                        <br />② 1着馬の予想順位 → {validation.winnerRank}位{" "}
                        {validation.winnerRank <= 3 ? "✅" : "（要検討）"}
                        <br />③ 温度パラメータの妥当性 → 1着確率{" "}
                        {formatPct(validation.winnerProb)}{" "}
                        {validation.winnerProb > 0.15
                          ? "（堅い決着向き）"
                          : "（混戦向き）"}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            )}
            {!validation && horses.length > 0 && (
              <div style={s.card}>
                <p style={{ fontSize: 12, color: "#6B7280" }}>
                  上のフォームで1〜3着の馬番を選択すると、検証結果が表示されます。
                </p>
              </div>
            )}
            {horses.length === 0 && (
              <div style={s.card}>
                <p style={{ fontSize: 12, color: "#6B7280" }}>
                  先にレースデータを入力してください。
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
