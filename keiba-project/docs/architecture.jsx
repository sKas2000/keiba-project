import { useState } from "react";

const PHASES = [
  {
    id: "p0",
    num: "0",
    title: "レース選定",
    timing: "毎週月曜",
    color: "#8B5CF6",
    roles: {
      human: "対象レースを選定。グレード・予算・興味で判断",
      claude: "当週の重賞・注目レース一覧を提示。コース特性の概要を提供",
      tool: "週間予算管理画面で残予算を表示",
    },
    inputs: ["JRAレースカレンダー", "月間予算残高"],
    outputs: ["週間レース計画（対象レース＋仮予算配分）"],
    gate: "期待値が追えないレースは最初から除外",
    status: "new",
    problems: [],
  },
  {
    id: "p1",
    num: "1",
    title: "データ収集・構造化",
    timing: "レース2-3日前",
    color: "#3B82F6",
    roles: {
      human: "出馬表データを提供（PDF/コピペ/URL）",
      claude: "データをパースし構造化。不足情報を検索で補完",
      tool: "構造化データをツールに自動入力",
    },
    inputs: ["出馬表PDF", "netkeiba出馬表", "過去走データ"],
    outputs: ["全馬の構造化データ（馬名・戦績・オッズ・騎手・枠順）"],
    gate: "データの欠損がないことを確認",
    status: "partial",
    problems: ["入力フォーマット未統一", "netkeiba自動取得不可"],
  },
  {
    id: "p2",
    num: "2",
    title: "評価点付与",
    timing: "レース2-3日前",
    color: "#06B6D4",
    roles: {
      human: "評価点を確認・修正。最終承認",
      claude: "100点満点で全馬を評価。根拠を明示。枠順補正・展開補正を適用",
      tool: "評価内訳を表示。点数は直接編集可能",
    },
    inputs: ["構造化データ", "コース特性", "ペース予想"],
    outputs: ["全馬の評価点（内訳付き）", "◎○▲△☆ランキング"],
    gate: "人間が評価点を確認し、納得してから次へ",
    status: "partial",
    problems: ["評価基準の主観性", "新馬・未勝利のGI実績代替基準が未整備"],
  },
  {
    id: "p3",
    num: "3",
    title: "確率変換",
    timing: "自動（評価確定後即座）",
    color: "#10B981",
    roles: {
      human: "温度パラメータを確認。必要に応じて調整",
      claude: "レース特性（堅い/荒れやすい）から温度を推奨",
      tool: "ソフトマックス変換で全馬の1着・3着内確率を算出。市場確率との乖離を表示",
    },
    inputs: ["評価点", "温度パラメータ"],
    outputs: ["全馬の1着確率", "全馬の3着内確率", "市場確率との差異分析"],
    gate: "モデル確率と市場確率の大幅乖離があれば評価点を再検討",
    status: "done",
    problems: [],
  },
  {
    id: "p4",
    num: "4",
    title: "オッズ取得・期待値計算",
    timing: "発走30-60分前",
    color: "#EAB308",
    roles: {
      human: "最新オッズを入力（または確認）",
      claude: "オッズの異常値や急変を指摘",
      tool: "全券種の的中確率・期待値を自動算出。S/A/B/Cランク付け",
    },
    inputs: ["確率データ", "最新オッズ（単勝・複勝・馬連・ワイド等）"],
    outputs: ["全券種の期待値ランキング", "EV+買い目リスト"],
    gate: "EV+の買い目が3点未満なら見送り検討",
    status: "partial",
    problems: ["組合せオッズの推定が粗い（積÷定数）", "実オッズ入力の手間"],
  },
  {
    id: "p5",
    num: "5",
    title: "購入判断・資金配分",
    timing: "発走15-30分前",
    color: "#F97316",
    roles: {
      human: "最終的な購入/見送りを決定。金額を確定",
      claude: "推奨買い目と配分を提示。リスク評価",
      tool: "予算内で期待値比例の配分を自動計算。100円単位に丸め",
    },
    inputs: ["EV+買い目リスト", "レース予算", "確信度"],
    outputs: ["具体的な買い目リスト（券種・組合せ・金額）"],
    gate: "予算超過チェック。確信度閾値チェック",
    status: "new",
    problems: ["見送り基準の閾値が未検証", "レース横断配分ロジック未実装"],
  },
  {
    id: "p6",
    num: "6",
    title: "結果記録・検証",
    timing: "レース後当日中",
    color: "#EF4444",
    roles: {
      human: "結果を入力",
      claude: "評価の妥当性を分析。改善点を特定",
      tool: "的中/不的中を自動判定。収支計算。評価精度メトリクスを更新",
    },
    inputs: ["着順結果", "払戻金"],
    outputs: ["検証レポート", "累積成績データ", "評価精度推移"],
    gate: "評価点上位3頭が3着内に入ったか",
    status: "partial",
    problems: ["検証データの蓄積・横断分析の仕組みがない"],
  },
  {
    id: "p7",
    num: "F",
    title: "フィードバック・改善",
    timing: "月次 or 10レースごと",
    color: "#EC4899",
    roles: {
      human: "改善方針を決定。指示書を更新",
      claude: "蓄積データから傾向分析。温度パラメータ最適値を提案",
      tool: "評価精度・回収率・確信度の推移グラフ。最適パラメータのバックテスト",
    },
    inputs: ["累積検証データ", "収支記録"],
    outputs: ["プロジェクト指示書の更新", "パラメータ調整", "評価基準の改訂"],
    gate: "10レース以上のデータが溜まってから大幅改訂",
    status: "new",
    problems: ["データ蓄積の仕組みが未構築"],
  },
];

const statusMap = {
  done: { label: "実装済", color: "#22C55E" },
  partial: { label: "一部実装", color: "#EAB308" },
  new: { label: "未着手", color: "#EF4444" },
};

export default function Architecture() {
  const [selected, setSelected] = useState("p2");

  const sel = PHASES.find((p) => p.id === selected);

  return (
    <div style={{ fontFamily: "'Noto Sans JP', sans-serif", background: "#08090C", color: "#D1D5DB", minHeight: "100vh", padding: "16px" }}>
      <div style={{ maxWidth: 960, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ borderBottom: "1px solid #1F2937", paddingBottom: 14, marginBottom: 20 }}>
          <h1 style={{ fontSize: 20, fontWeight: 700, color: "#F9FAFB", margin: 0 }}>
            競馬分析プロジェクト v3.0 アーキテクチャ
          </h1>
          <p style={{ fontSize: 11, color: "#6B7280", marginTop: 4 }}>
            データ入力 → 評価 → 確率 → 期待値 → 購入判断 → 検証 の全パイプライン設計
          </p>
        </div>

        {/* Pipeline flow */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 10, color: "#6B7280", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            パイプライン全体像
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 3, flexWrap: "wrap" }}>
            {PHASES.map((p, i) => {
              const st = statusMap[p.status];
              const isSel = selected === p.id;
              return (
                <div key={p.id} style={{ display: "flex", alignItems: "center", gap: 3 }}>
                  <button
                    onClick={() => setSelected(p.id)}
                    style={{
                      background: isSel ? p.color + "20" : "#111318",
                      border: `1.5px solid ${isSel ? p.color : "#1F2937"}`,
                      borderRadius: 8,
                      padding: "8px 10px",
                      cursor: "pointer",
                      color: isSel ? "#F9FAFB" : "#9CA3AF",
                      fontSize: 11,
                      fontWeight: isSel ? 600 : 400,
                      transition: "all 0.15s",
                      minWidth: 80,
                      textAlign: "center",
                      position: "relative",
                    }}
                  >
                    <div style={{ fontSize: 9, color: p.color, marginBottom: 2 }}>Phase {p.num}</div>
                    <div>{p.title}</div>
                    <div style={{ position: "absolute", top: -6, right: -4 }}>
                      <span style={{
                        fontSize: 7, padding: "1px 4px", borderRadius: 3,
                        background: st.color + "20", color: st.color,
                        border: `1px solid ${st.color}40`,
                      }}>{st.label}</span>
                    </div>
                  </button>
                  {i < PHASES.length - 1 && (
                    <span style={{ color: "#2F3336", fontSize: 14 }}>→</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected phase detail */}
        {sel && (
          <div style={{ border: `1px solid ${sel.color}40`, borderRadius: 12, overflow: "hidden", marginBottom: 20 }}>
            <div style={{ background: sel.color + "15", padding: "14px 18px", borderBottom: `1px solid ${sel.color}30` }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div>
                  <span style={{ fontSize: 11, color: sel.color, fontWeight: 600 }}>Phase {sel.num}</span>
                  <h2 style={{ fontSize: 17, fontWeight: 700, color: "#F9FAFB", margin: "2px 0 0" }}>{sel.title}</h2>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 10, color: "#6B7280" }}>タイミング</div>
                  <div style={{ fontSize: 12, color: "#E5E7EB" }}>{sel.timing}</div>
                </div>
              </div>
            </div>

            {/* Role cards */}
            <div style={{ padding: "14px 18px" }}>
              <div style={{ fontSize: 11, color: "#6B7280", marginBottom: 8, fontWeight: 600 }}>役割分担</div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 16 }}>
                {[
                  { key: "human", label: "🧑 人間", color: "#F97316" },
                  { key: "claude", label: "🤖 Claude", color: "#3B82F6" },
                  { key: "tool", label: "⚙️ ツール", color: "#10B981" },
                ].map((r) => (
                  <div key={r.key} style={{
                    background: "#0B0E11", borderRadius: 8, padding: 10,
                    border: `1px solid ${r.color}30`,
                  }}>
                    <div style={{ fontSize: 10, color: r.color, fontWeight: 600, marginBottom: 4 }}>{r.label}</div>
                    <div style={{ fontSize: 11, color: "#D1D5DB", lineHeight: 1.6 }}>{sel.roles[r.key]}</div>
                  </div>
                ))}
              </div>

              {/* IO */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 16 }}>
                <div>
                  <div style={{ fontSize: 10, color: "#6B7280", fontWeight: 600, marginBottom: 4 }}>入力</div>
                  {sel.inputs.map((inp, i) => (
                    <div key={i} style={{ fontSize: 11, color: "#9CA3AF", padding: "2px 0" }}>→ {inp}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "#6B7280", fontWeight: 600, marginBottom: 4 }}>出力</div>
                  {sel.outputs.map((out, i) => (
                    <div key={i} style={{ fontSize: 11, color: "#E5E7EB", padding: "2px 0", fontWeight: 500 }}>← {out}</div>
                  ))}
                </div>
              </div>

              {/* Quality gate */}
              <div style={{
                background: "#1E293B", borderRadius: 6, padding: "8px 12px",
                border: "1px solid #334155", marginBottom: 12,
              }}>
                <div style={{ fontSize: 10, color: "#EAB308", fontWeight: 600, marginBottom: 2 }}>品質ゲート</div>
                <div style={{ fontSize: 11, color: "#E5E7EB" }}>{sel.gate}</div>
              </div>

              {/* Problems */}
              {sel.problems.length > 0 && (
                <div>
                  <div style={{ fontSize: 10, color: "#EF4444", fontWeight: 600, marginBottom: 4 }}>未解決の課題</div>
                  {sel.problems.map((p, i) => (
                    <div key={i} style={{ fontSize: 11, color: "#FCA5A5", padding: "2px 0" }}>• {p}</div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Design principles */}
        <div style={{ background: "#111318", border: "1px solid #1F2937", borderRadius: 10, padding: 16, marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#E5E7EB", marginBottom: 10 }}>設計原則</div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[
              { title: "人間 = 判断", desc: "評価の最終承認、購入/見送りの決定、予算管理は人間が行う。ツールやClaudeは判断材料を提供するだけ" },
              { title: "Claude = 分析", desc: "馬の評価、ペース予想、レース特性分析など、定性的な判断を伴う分析を担当" },
              { title: "ツール = 計算", desc: "確率変換、期待値計算、資金配分など、数学的な計算と大量の組合せ処理を担当" },
              { title: "品質ゲート = 安全弁", desc: "各フェーズの出力が次のフェーズに渡る前にチェック。ゴミが入ればゴミが出る(GIGO)を防ぐ" },
            ].map((p, i) => (
              <div key={i} style={{ background: "#0B0E11", borderRadius: 6, padding: 10, border: "1px solid #1F2937" }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "#F9FAFB", marginBottom: 4 }}>{p.title}</div>
                <div style={{ fontSize: 11, color: "#9CA3AF", lineHeight: 1.6 }}>{p.desc}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Priority roadmap */}
        <div style={{ background: "#111318", border: "1px solid #1F2937", borderRadius: 10, padding: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 600, color: "#E5E7EB", marginBottom: 10 }}>実装ロードマップ</div>
          {[
            { phase: "完了", items: ["確率変換エンジン（ソフトマックス）", "単勝/複勝/馬連/ワイド/3連複の的中率自動算出", "期待値ランキング表示", "市場確率との乖離分析"], color: "#22C55E" },
            { phase: "優先度1", items: ["組合せオッズの推定精度改善（実オッズ入力対応）", "評価基準の定量化（未勝利戦・条件戦用の基準整備）"], color: "#EF4444" },
            { phase: "優先度2", items: ["見送り判断の閾値検証（10レース分のデータ蓄積後）", "レース横断の資金配分ロジック（ケリー基準ベース）"], color: "#F97316" },
            { phase: "優先度3", items: ["検証データの蓄積・可視化", "温度パラメータのバックテスト", "データ入力の効率化（フォーマット統一）"], color: "#EAB308" },
            { phase: "優先度4", items: ["プロジェクト指示書v3.0の正式化", "月次レビュープロセスの確立"], color: "#6B7280" },
          ].map((r, i) => (
            <div key={i} style={{ display: "flex", gap: 10, marginBottom: 10, alignItems: "flex-start" }}>
              <div style={{
                fontSize: 9, fontWeight: 700, color: r.color,
                background: r.color + "15", border: `1px solid ${r.color}40`,
                borderRadius: 4, padding: "2px 8px", whiteSpace: "nowrap", minWidth: 60, textAlign: "center",
                marginTop: 2,
              }}>{r.phase}</div>
              <div style={{ fontSize: 11, color: "#D1D5DB", lineHeight: 1.7 }}>
                {r.items.join(" → ")}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
