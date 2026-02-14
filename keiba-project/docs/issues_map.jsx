import { useState } from "react";

const phases = [
  {
    id: "step1",
    phase: "Step 1",
    title: "事前分析（実力評価）",
    status: "partial",
    problems: [
      {
        severity: "medium",
        title: "評価点の主観性",
        detail:
          "100点満点の配点自体は整備されているが、各馬への点数付けはClaudeの主観に依存。同じ馬でもチャットごとに点数がブレる可能性がある。",
        impact: "後続の全計算の土台が不安定になる",
        current: "Claudeが都度判断",
        ideal: "過去成績の定量指標（スピード指数・レーティング等）をベースに自動算出し、主観補正は限定的に",
      },
      {
        severity: "low",
        title: "ペース予想の精度",
        detail:
          "全馬の脚質を考慮したペース予想は指示書に記載があるが、「ハイ・ミドル・スロー」の3分類が粗い。各馬の位置取りまで想定する手順が曖昧。",
        impact: "展開補正（Step 3.5）の精度に直結",
        current: "3分類のペース予想",
        ideal: "各馬のポジション想定→隊列図→ペース数値化",
      },
    ],
  },
  {
    id: "step2",
    phase: "Step 2",
    title: "枠順補正",
    status: "ok",
    problems: [
      {
        severity: "low",
        title: "補正ルールは整備済み",
        detail:
          "±5点以内、超一流騎手・実力馬の減額条件など、小倉牝馬Sの教訓が反映されている。大きな問題はない。",
        impact: "—",
        current: "ルール通り運用",
        ideal: "コース別の枠順データを蓄積して補正値を微調整",
      },
    ],
  },
  {
    id: "step3",
    phase: "Step 3",
    title: "最終評価（オッズ確定後）",
    status: "problem",
    problems: [
      {
        severity: "high",
        title: "評価点→確率変換のロジックがない",
        detail:
          "評価点ランキングは出るが「この馬の1着確率は何%か」を算出する仕組みがない。点数は序列を示すが確率ではない。85点の馬が75点の馬より何倍強いのかが不明。",
        impact:
          "期待値計算の根幹が欠落。以降の全ステップが「感覚的な確率」に依存する",
        current: "評価点ランキング止まり",
        ideal:
          "ソフトマックス関数等で評価点を確率に変換。温度パラメータでレースの荒れやすさを調整",
      },
      {
        severity: "high",
        title: "オッズからの期待値計算が手動",
        detail:
          "「期待値120円以上を買い推奨」と書いてあるが、全券種の期待値を手計算するのは18頭立てでは非現実的。馬連153通り、3連複816通り。",
        impact: "実際には一部の買い目しか検討できず、最適な馬券を見逃す",
        current: "主要な数点のみ手計算",
        ideal: "全券種の期待値を自動計算し、上位をリスト表示",
      },
    ],
  },
  {
    id: "step35",
    phase: "Step 3.5",
    title: "展開補正",
    status: "partial",
    problems: [
      {
        severity: "medium",
        title: "補正値の定量基準が曖昧",
        detail:
          "「直線が短い＋ハイペース→先行有利大」とあるが、「大」が何点なのか明示されていない。Claudeの裁量に依存。",
        impact: "補正の再現性が低い",
        current: "定性的なガイドライン",
        ideal:
          "コース×ペース×脚質のマトリクスで補正値を定義（例：小倉芝1800m＋ハイペース＋追込＝-8点）",
      },
    ],
  },
  {
    id: "step36",
    phase: "Step 3.6",
    title: "馬券購入理論",
    status: "problem",
    problems: [
      {
        severity: "critical",
        title: "着順パターン10-15個では全体をカバーできない",
        detail:
          "18頭立ての1-2-3着組み合わせは4,896通り。15パターンでカバーできるのは全体の0.3%。残り99.7%が「その他」に押し込まれ、確率配分が歪む。",
        impact:
          "的中率の計算が構造的に不正確。期待値計算の信頼性が根本から崩れる",
        current: "手動で15パターン作成",
        ideal:
          "馬ごとの着順確率から条件付き確率で全券種の的中率を数学的に算出",
      },
      {
        severity: "critical",
        title: "パターン確率の根拠が薄い",
        detail:
          "「パターンA: 15%、B: 12%」のように確率を割り振るが、この数値自体が感覚的。「合計100%に調整」という作業が恣意性を増幅させる。",
        impact: "期待値計算の入力値が信頼できない",
        current: "Claudeが感覚で割り振り",
        ideal: "Step 3の確率変換結果から自動算出",
      },
      {
        severity: "high",
        title: "期待値ランクの基準が固定的",
        detail:
          "S級1,000円以上、A級400-999円等の基準が全レース共通。しかしG1（予算1万円）とG3（予算3千円）では同じ期待値でも意味が異なる。",
        impact: "レースの格や予算に応じた柔軟な判断ができない",
        current: "固定ランク",
        ideal: "予算・レース格に応じた動的な閾値設定",
      },
    ],
  },
  {
    id: "betting",
    phase: "購入判断",
    title: "買う／買わないの判断",
    status: "problem",
    problems: [
      {
        severity: "high",
        title: "「見送り」の判断基準がない",
        detail:
          "指示書には「期待値120円以上を買い推奨」とあるが、レース全体の期待値が低い場合に「このレースは見送る」という判断基準が存在しない。",
        impact:
          "期待値の低いレースにも予算を使ってしまい、長期的な回収率が下がる",
        current: "対象レースは原則すべて買う",
        ideal:
          "レース全体の確信度を算出し、閾値以下なら見送り。見送り判断も記録して検証",
      },
      {
        severity: "high",
        title: "資金配分がレース横断で最適化されていない",
        detail:
          "各レースの予算はG1=1万、G2=5千、G3=3千と固定。しかし「自信のあるG3」と「自信のないG1」では配分を変えるべき。",
        impact: "期待値の高いレースに資金を集中できない",
        current: "グレード別の固定予算",
        ideal:
          "ケリー基準等に基づき、確信度×期待値でレース間の資金配分を動的に決定",
      },
      {
        severity: "medium",
        title: "時間制約への対応が不十分",
        detail:
          "Step 3は「発走30分前」に実施するが、オッズ確認→期待値計算→買い目決定→購入を30分で完了するのは現実的に厳しい。特にStep 3.6の手計算は時間がかかる。",
        impact: "時間切れで不完全な分析のまま購入、または購入機会の逸失",
        current: "30分で全て手動実行",
        ideal:
          "事前にオッズシナリオを準備し、確定オッズを入力するだけで買い目が出る仕組み",
      },
    ],
  },
];

const severityConfig = {
  critical: {
    label: "致命的",
    color: "#DC2626",
    bg: "#FEF2F2",
    border: "#FECACA",
  },
  high: { label: "重大", color: "#EA580C", bg: "#FFF7ED", border: "#FED7AA" },
  medium: { label: "中程度", color: "#CA8A04", bg: "#FEFCE8", border: "#FEF08A" },
  low: { label: "軽微", color: "#16A34A", bg: "#F0FDF4", border: "#BBF7D0" },
};

const statusConfig = {
  ok: { label: "問題少", color: "#16A34A", icon: "○" },
  partial: { label: "改善余地", color: "#CA8A04", icon: "△" },
  problem: { label: "要改善", color: "#DC2626", icon: "✕" },
};

export default function IssuesMap() {
  const [expandedPhase, setExpandedPhase] = useState("step36");
  const [showDetail, setShowDetail] = useState(null);

  const totalProblems = phases.reduce((s, p) => s + p.problems.length, 0);
  const criticalCount = phases.reduce(
    (s, p) => s + p.problems.filter((pr) => pr.severity === "critical").length,
    0
  );
  const highCount = phases.reduce(
    (s, p) => s + p.problems.filter((pr) => pr.severity === "high").length,
    0
  );

  return (
    <div
      style={{
        fontFamily: "'Noto Sans JP', 'Hiragino Sans', sans-serif",
        background: "#0F1419",
        color: "#E7E9EA",
        minHeight: "100vh",
        padding: "24px",
      }}
    >
      {/* Header */}
      <div style={{ maxWidth: 880, margin: "0 auto" }}>
        <div
          style={{
            borderBottom: "1px solid #2F3336",
            paddingBottom: 20,
            marginBottom: 28,
          }}
        >
          <h1
            style={{
              fontSize: 22,
              fontWeight: 700,
              margin: 0,
              letterSpacing: "-0.02em",
            }}
          >
            馬券購入プロセス — 問題点マップ
          </h1>
          <p
            style={{
              color: "#71767B",
              fontSize: 13,
              margin: "8px 0 16px",
              lineHeight: 1.5,
            }}
          >
            分析開始から馬券購入までの各フェーズで、どこにボトルネックがあるかを整理
          </p>

          {/* Summary stats */}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div
              style={{
                background: "#1A1F25",
                borderRadius: 8,
                padding: "10px 16px",
                border: "1px solid #2F3336",
              }}
            >
              <span style={{ color: "#71767B", fontSize: 11 }}>総問題数</span>
              <div style={{ fontSize: 22, fontWeight: 700 }}>
                {totalProblems}
              </div>
            </div>
            <div
              style={{
                background: "#1A1F25",
                borderRadius: 8,
                padding: "10px 16px",
                border: "1px solid #2F3336",
              }}
            >
              <span style={{ color: "#DC2626", fontSize: 11 }}>致命的</span>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#DC2626" }}>
                {criticalCount}
              </div>
            </div>
            <div
              style={{
                background: "#1A1F25",
                borderRadius: 8,
                padding: "10px 16px",
                border: "1px solid #2F3336",
              }}
            >
              <span style={{ color: "#EA580C", fontSize: 11 }}>重大</span>
              <div style={{ fontSize: 22, fontWeight: 700, color: "#EA580C" }}>
                {highCount}
              </div>
            </div>
          </div>
        </div>

        {/* Flow diagram */}
        <div style={{ marginBottom: 32 }}>
          <div
            style={{
              fontSize: 12,
              color: "#71767B",
              marginBottom: 12,
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            ワークフロー全体像
          </div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              flexWrap: "wrap",
            }}
          >
            {phases.map((p, i) => {
              const st = statusConfig[p.status];
              return (
                <div
                  key={p.id}
                  style={{ display: "flex", alignItems: "center", gap: 4 }}
                >
                  <button
                    onClick={() =>
                      setExpandedPhase(expandedPhase === p.id ? null : p.id)
                    }
                    style={{
                      background:
                        expandedPhase === p.id ? "#1D2A3A" : "#1A1F25",
                      border: `1px solid ${expandedPhase === p.id ? "#3B82F6" : "#2F3336"}`,
                      borderRadius: 8,
                      padding: "8px 12px",
                      cursor: "pointer",
                      color: "#E7E9EA",
                      fontSize: 12,
                      fontWeight: expandedPhase === p.id ? 600 : 400,
                      transition: "all 0.15s",
                      whiteSpace: "nowrap",
                    }}
                  >
                    <span style={{ color: st.color, marginRight: 4 }}>
                      {st.icon}
                    </span>
                    {p.phase}
                  </button>
                  {i < phases.length - 1 && (
                    <span style={{ color: "#2F3336", fontSize: 16 }}>→</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Phase details */}
        {phases.map((phase) => {
          const isExpanded = expandedPhase === phase.id;
          const st = statusConfig[phase.status];

          return (
            <div
              key={phase.id}
              style={{
                marginBottom: 12,
                border: `1px solid ${isExpanded ? "#3B82F6" : "#2F3336"}`,
                borderRadius: 12,
                overflow: "hidden",
                transition: "border-color 0.15s",
              }}
            >
              {/* Phase header */}
              <button
                onClick={() =>
                  setExpandedPhase(isExpanded ? null : phase.id)
                }
                style={{
                  width: "100%",
                  background: isExpanded ? "#1D2A3A" : "#1A1F25",
                  border: "none",
                  padding: "14px 18px",
                  cursor: "pointer",
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  color: "#E7E9EA",
                  transition: "background 0.15s",
                }}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                  }}
                >
                  <span
                    style={{
                      color: st.color,
                      fontWeight: 700,
                      fontSize: 14,
                      width: 20,
                    }}
                  >
                    {st.icon}
                  </span>
                  <div style={{ textAlign: "left" }}>
                    <div style={{ fontWeight: 600, fontSize: 14 }}>
                      {phase.phase}: {phase.title}
                    </div>
                    <div style={{ color: "#71767B", fontSize: 12, marginTop: 2 }}>
                      問題 {phase.problems.length}件
                      {phase.problems.some((p) => p.severity === "critical") &&
                        " — 致命的あり"}
                    </div>
                  </div>
                </div>
                <span
                  style={{
                    color: "#71767B",
                    fontSize: 18,
                    transform: isExpanded ? "rotate(180deg)" : "none",
                    transition: "transform 0.2s",
                  }}
                >
                  ▾
                </span>
              </button>

              {/* Problems list */}
              {isExpanded && (
                <div style={{ padding: "4px 18px 18px" }}>
                  {phase.problems.map((problem, pi) => {
                    const sev = severityConfig[problem.severity];
                    const detailKey = `${phase.id}-${pi}`;
                    const isDetailOpen = showDetail === detailKey;

                    return (
                      <div
                        key={pi}
                        style={{
                          background: "#0F1419",
                          border: `1px solid ${sev.border}33`,
                          borderRadius: 10,
                          marginTop: 10,
                          overflow: "hidden",
                        }}
                      >
                        <button
                          onClick={() =>
                            setShowDetail(isDetailOpen ? null : detailKey)
                          }
                          style={{
                            width: "100%",
                            background: "transparent",
                            border: "none",
                            padding: "12px 14px",
                            cursor: "pointer",
                            color: "#E7E9EA",
                            display: "flex",
                            alignItems: "flex-start",
                            gap: 10,
                            textAlign: "left",
                          }}
                        >
                          <span
                            style={{
                              fontSize: 10,
                              fontWeight: 700,
                              color: sev.color,
                              background: `${sev.color}18`,
                              border: `1px solid ${sev.color}40`,
                              borderRadius: 4,
                              padding: "2px 6px",
                              whiteSpace: "nowrap",
                              flexShrink: 0,
                              marginTop: 1,
                            }}
                          >
                            {sev.label}
                          </span>
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 600, fontSize: 13 }}>
                              {problem.title}
                            </div>
                            <div
                              style={{
                                color: "#71767B",
                                fontSize: 12,
                                marginTop: 4,
                                lineHeight: 1.5,
                              }}
                            >
                              {problem.detail}
                            </div>
                          </div>
                          <span
                            style={{
                              color: "#71767B",
                              fontSize: 14,
                              flexShrink: 0,
                              transform: isDetailOpen
                                ? "rotate(90deg)"
                                : "none",
                              transition: "transform 0.15s",
                            }}
                          >
                            ▸
                          </span>
                        </button>

                        {isDetailOpen && (
                          <div
                            style={{
                              padding: "0 14px 14px",
                              borderTop: "1px solid #2F3336",
                              marginTop: 0,
                            }}
                          >
                            <div
                              style={{
                                display: "grid",
                                gridTemplateColumns: "1fr 1fr",
                                gap: 10,
                                marginTop: 12,
                              }}
                            >
                              <div>
                                <div
                                  style={{
                                    fontSize: 10,
                                    color: "#EA580C",
                                    fontWeight: 600,
                                    marginBottom: 4,
                                    textTransform: "uppercase",
                                    letterSpacing: "0.05em",
                                  }}
                                >
                                  影響
                                </div>
                                <div
                                  style={{
                                    fontSize: 12,
                                    color: "#A1A5AA",
                                    lineHeight: 1.5,
                                  }}
                                >
                                  {problem.impact}
                                </div>
                              </div>
                              <div>
                                <div
                                  style={{
                                    fontSize: 10,
                                    color: "#71767B",
                                    fontWeight: 600,
                                    marginBottom: 4,
                                    textTransform: "uppercase",
                                    letterSpacing: "0.05em",
                                  }}
                                >
                                  現状
                                </div>
                                <div
                                  style={{
                                    fontSize: 12,
                                    color: "#A1A5AA",
                                    lineHeight: 1.5,
                                  }}
                                >
                                  {problem.current}
                                </div>
                              </div>
                            </div>
                            <div style={{ marginTop: 10 }}>
                              <div
                                style={{
                                  fontSize: 10,
                                  color: "#3B82F6",
                                  fontWeight: 600,
                                  marginBottom: 4,
                                  textTransform: "uppercase",
                                  letterSpacing: "0.05em",
                                }}
                              >
                                理想形
                              </div>
                              <div
                                style={{
                                  fontSize: 12,
                                  color: "#93C5FD",
                                  lineHeight: 1.5,
                                  background: "#1E293B",
                                  padding: "8px 10px",
                                  borderRadius: 6,
                                }}
                              >
                                {problem.ideal}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}

        {/* Conclusion */}
        <div
          style={{
            marginTop: 28,
            background: "#1A1F25",
            border: "1px solid #2F3336",
            borderRadius: 12,
            padding: 20,
          }}
        >
          <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 12 }}>
            構造的な結論
          </div>
          <div style={{ fontSize: 13, lineHeight: 1.7, color: "#A1A5AA" }}>
            問題の核心は
            <span style={{ color: "#E7E9EA", fontWeight: 600 }}>
              「評価点→確率→期待値」の変換パイプラインが断絶
            </span>
            していること。Step 1-2で作った評価点ランキングが、Step
            3.6の馬券購入判断に数学的に接続されていない。
            間を埋めているのはClaudeの感覚的な確率割り振りであり、
            これがシステム全体の信頼性を下げている。
          </div>
          <div
            style={{
              marginTop: 14,
              padding: "12px 14px",
              background: "#0F1419",
              borderRadius: 8,
              border: "1px solid #2F3336",
            }}
          >
            <div
              style={{
                fontSize: 11,
                color: "#3B82F6",
                fontWeight: 600,
                marginBottom: 6,
              }}
            >
              優先的に解決すべき2点
            </div>
            <div style={{ fontSize: 13, lineHeight: 1.7 }}>
              <span style={{ color: "#DC2626", fontWeight: 600 }}>①</span>{" "}
              評価点→確率の変換ロジック確立（全券種の的中率が自動算出可能に）
              <br />
              <span style={{ color: "#EA580C", fontWeight: 600 }}>②</span>{" "}
              レース単位の見送り判断＋レース横断の資金配分ロジック
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
