import { useState } from "react";

const VIEWS = ["全体フロー", "データの流れ", "ファイル構成"];

// Status indicators
const Status = ({ type }) => {
  const map = {
    ok: { bg: "#0D3B2E", border: "#166534", color: "#4ADE80", label: "稼働中" },
    broken: { bg: "#3B1318", border: "#7F1D1D", color: "#F87171", label: "要改修" },
    missing: { bg: "#3B2308", border: "#7C4A16", color: "#FBBF24", label: "未実装" },
    manual: { bg: "#1E2A3A", border: "#1E40AF", color: "#60A5FA", label: "手動" },
  };
  const s = map[type] || map.ok;
  return (
    <span style={{
      fontSize: 9, fontWeight: 700, padding: "2px 7px", borderRadius: 3,
      background: s.bg, border: `1px solid ${s.border}`, color: s.color,
      letterSpacing: "0.03em",
    }}>{s.label}</span>
  );
};

// Arrow connector
const Arrow = ({ label, broken }) => (
  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "6px 0" }}>
    <div style={{
      width: 2, height: 20,
      background: broken ? "#7F1D1D" : "#374151",
      borderLeft: broken ? "2px dashed #EF4444" : "none",
    }} />
    {label && (
      <div style={{
        fontSize: 9, color: broken ? "#F87171" : "#6B7280",
        padding: "1px 6px", borderRadius: 3,
        background: broken ? "#3B131820" : "transparent",
        fontWeight: broken ? 600 : 400,
      }}>{label}</div>
    )}
    <div style={{
      width: 0, height: 0,
      borderLeft: "5px solid transparent", borderRight: "5px solid transparent",
      borderTop: `6px solid ${broken ? "#EF4444" : "#374151"}`,
    }} />
  </div>
);

// Step card
const StepCard = ({ num, title, who, when, tool, status, children, highlight }) => (
  <div style={{
    background: highlight ? "#111827" : "#0D1117",
    border: `1px solid ${highlight ? "#3B82F6" : "#1F2937"}`,
    borderRadius: 10, padding: "14px 16px",
    boxShadow: highlight ? "0 0 20px #3B82F620" : "none",
    position: "relative",
  }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{
          fontSize: 11, fontWeight: 800, color: "#1F2937",
          background: highlight ? "#3B82F6" : "#374151",
          width: 24, height: 24, borderRadius: 12,
          display: "flex", alignItems: "center", justifyContent: "center",
          color: "#F9FAFB",
        }}>{num}</span>
        <span style={{ fontSize: 14, fontWeight: 700, color: "#F9FAFB" }}>{title}</span>
      </div>
      <Status type={status} />
    </div>
    <div style={{ display: "flex", gap: 16, marginBottom: 8, flexWrap: "wrap" }}>
      {who && (
        <div style={{ fontSize: 10, color: "#9CA3AF" }}>
          <span style={{ color: "#6B7280" }}>実行:</span> <span style={{ color: "#E5E7EB", fontWeight: 600 }}>{who}</span>
        </div>
      )}
      {when && (
        <div style={{ fontSize: 10, color: "#9CA3AF" }}>
          <span style={{ color: "#6B7280" }}>時期:</span> {when}
        </div>
      )}
      {tool && (
        <div style={{ fontSize: 10, color: "#9CA3AF" }}>
          <span style={{ color: "#6B7280" }}>ツール:</span> <span style={{ color: "#60A5FA" }}>{tool}</span>
        </div>
      )}
    </div>
    <div style={{ fontSize: 11, color: "#9CA3AF", lineHeight: 1.7 }}>
      {children}
    </div>
  </div>
);

// Data box
const DataBox = ({ label, items, color = "#3B82F6" }) => (
  <div style={{
    background: color + "08", border: `1px solid ${color}25`,
    borderRadius: 6, padding: "8px 10px",
  }}>
    <div style={{ fontSize: 10, fontWeight: 700, color, marginBottom: 4 }}>{label}</div>
    {items.map((item, i) => (
      <div key={i} style={{ fontSize: 10, color: "#D1D5DB", lineHeight: 1.6 }}>{item}</div>
    ))}
  </div>
);

// File tree item
const FileItem = ({ name, desc, status, indent = 0 }) => (
  <div style={{
    display: "flex", justifyContent: "space-between", alignItems: "center",
    padding: "4px 0 4px " + (indent * 16 + 8) + "px",
    borderBottom: "1px solid #111827",
  }}>
    <div>
      <span style={{
        fontSize: 11, color: "#E5E7EB", fontFamily: "'JetBrains Mono', monospace",
        fontWeight: 500,
      }}>{name}</span>
      {desc && <span style={{ fontSize: 10, color: "#6B7280", marginLeft: 8 }}>{desc}</span>}
    </div>
    {status && <Status type={status} />}
  </div>
);

export default function FlowDiagram() {
  const [view, setView] = useState("全体フロー");

  return (
    <div style={{
      fontFamily: "'Noto Sans JP', 'Hiragino Sans', system-ui, sans-serif",
      background: "#08090C", color: "#D1D5DB", minHeight: "100vh", padding: "20px",
    }}>
      <div style={{ maxWidth: 800, margin: "0 auto" }}>
        {/* Header */}
        <div style={{ borderBottom: "1px solid #1F2937", paddingBottom: 14, marginBottom: 20 }}>
          <h1 style={{ fontSize: 18, fontWeight: 800, color: "#F9FAFB", margin: 0, letterSpacing: "-0.02em" }}>
            競馬分析プロジェクト 構成図
          </h1>
          <p style={{ fontSize: 11, color: "#6B7280", marginTop: 4 }}>
            v1.4スクレイパー + ev_calculator の接続設計
          </p>
        </div>

        {/* View tabs */}
        <div style={{ display: "flex", gap: 2, marginBottom: 20, borderBottom: "1px solid #1F2937" }}>
          {VIEWS.map(v => (
            <button key={v} onClick={() => setView(v)} style={{
              padding: "8px 16px", fontSize: 12, fontWeight: view === v ? 700 : 400,
              color: view === v ? "#60A5FA" : "#6B7280",
              background: "transparent", border: "none",
              borderBottom: view === v ? "2px solid #60A5FA" : "2px solid transparent",
              cursor: "pointer",
            }}>{v}</button>
          ))}
        </div>

        {/* =================== 全体フロー =================== */}
        {view === "全体フロー" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 0, alignItems: "stretch" }}>

            <StepCard num="1" title="レース選定" who="春日" when="当日午前" status="manual">
              対象レースを決定（グレード・予算・興味）
            </StepCard>
            <Arrow label="対象レースが決まる" />

            <StepCard num="2" title="オッズ取得" who="春日（PC）" when="発走1-2h前" tool="jra_scraper.py v1.4" status="ok">
              <div style={{ color: "#4ADE80" }}>✅ ターミナルで実行 → input.json 自動生成</div>
              <div>全馬の単勝・複勝 + 馬連120組 + ワイド120組 + 3連複560組</div>
              <div style={{ color: "#FBBF24", marginTop: 4 }}>⚠️ 馬単240組も取得中（不要→削除予定）</div>
            </StepCard>
            <Arrow label="input.json" />

            <StepCard num="3" title="評価点付与" who="Claude" when="Step 2直後" tool="このチャット" status="ok">
              <div>input.json をアップロード →「Phase 2お願い」</div>
              <div>全馬100点満点評価 + 枠順補正 + 展開補正 + 本命宣言</div>
            </StepCard>
            <Arrow label="評価点リスト（テキスト）" />

            <StepCard num="4" title="評価点の承認" who="春日" when="Step 3直後" status="manual">
              <div>Claudeの評価を確認 → OK or 修正指示</div>
              <div style={{ color: "#F87171" }}>品質ゲート: 承認なしに先に進まない</div>
            </StepCard>
            <Arrow label="確定した評価点" broken />

            <StepCard num="5" title="期待値計算" who="春日" when="発走30-60分前" tool="ev_calculator.jsx" status="broken" highlight>
              <div style={{ color: "#F87171", fontWeight: 700 }}>❌ ボトルネック: ここが断絶</div>
              <div style={{ marginTop: 6, padding: "6px 8px", background: "#3B131820", borderRadius: 4, border: "1px solid #7F1D1D40" }}>
                <div style={{ fontSize: 10, color: "#F87171", fontWeight: 600, marginBottom: 2 }}>現状の問題:</div>
                <div>① input.jsonを読み込めない（サンプルデータ固定）</div>
                <div>② 16頭の馬名・評価点・オッズを手で打ち直す</div>
                <div>③ 馬連・ワイド・3連複が推定オッズ（積÷定数）</div>
              </div>
              <div style={{ marginTop: 6, padding: "6px 8px", background: "#0D3B2E20", borderRadius: 4, border: "1px solid #16653440" }}>
                <div style={{ fontSize: 10, color: "#4ADE80", fontWeight: 600, marginBottom: 2 }}>v2で解消:</div>
                <div>① JSONドラッグ＆ドロップ → 全データ自動読込</div>
                <div>② 評価点だけ調整すればOK</div>
                <div>③ 実オッズ800組で正確な期待値</div>
              </div>
            </StepCard>
            <Arrow label="S/A級の買い目リスト" />

            <StepCard num="6" title="購入判断・実行" who="春日" when="発走15-30分前" status="manual">
              <div>期待値ランキングを見て購入 or 見送り</div>
              <div>確信度15%未満 or EV+3点未満 → 見送り検討</div>
            </StepCard>
            <Arrow label="レース結果" />

            <StepCard num="7" title="結果記録・検証" who="Claude + 春日" when="レース後当日" status="ok">
              <div>結果をChatに報告 → 検証レポート → result.json → GitHub</div>
              <div>評価点上位3頭が3着内に入ったかを最初に確認</div>
            </StepCard>

            {/* Legend */}
            <div style={{
              marginTop: 20, padding: 12, background: "#111827",
              borderRadius: 8, border: "1px solid #1F2937",
              display: "flex", gap: 16, flexWrap: "wrap",
            }}>
              <div style={{ fontSize: 10, color: "#6B7280", fontWeight: 600 }}>凡例:</div>
              <Status type="ok" />
              <Status type="broken" />
              <Status type="missing" />
              <Status type="manual" />
            </div>
          </div>
        )}

        {/* =================== データの流れ =================== */}
        {view === "データの流れ" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* JRA → Scraper */}
            <div style={{
              background: "#111827", borderRadius: 10, padding: 16,
              border: "1px solid #1F2937",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#F9FAFB", marginBottom: 10 }}>
                ① JRA公式 → スクレイパー → input.json
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 40px 1fr 40px 1fr", alignItems: "center", gap: 0 }}>
                <DataBox label="JRA公式サイト" color="#EF4444" items={[
                  "単勝・複勝オッズ",
                  "馬連三角行列",
                  "ワイド三角行列",
                  "3連複 二重三角行列",
                  "レース情報(race_title)",
                ]} />
                <div style={{ textAlign: "center", color: "#374151", fontSize: 18 }}>→</div>
                <DataBox label="jra_scraper.py v1.4" color="#3B82F6" items={[
                  "Playwright でページ操作",
                  "HTML解析・三角行列展開",
                  "race_title パース",
                  "JSON構造化",
                ]} />
                <div style={{ textAlign: "center", color: "#374151", fontSize: 18 }}>→</div>
                <DataBox label="input.json" color="#10B981" items={[
                  "race: レース情報",
                  "parameters: T, budget",
                  "horses[16]: 馬情報+オッズ",
                  "combo_odds.quinella[120]",
                  "combo_odds.wide[120]",
                  "combo_odds.trio[560]",
                  "horses[].score = 0（空）",
                ]} />
              </div>
            </div>

            {/* Claude fills scores */}
            <div style={{
              background: "#111827", borderRadius: 10, padding: 16,
              border: "1px solid #1F2937",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#F9FAFB", marginBottom: 10 }}>
                ② Claude が評価点を埋める
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 40px 1fr 40px 1fr", alignItems: "center", gap: 0 }}>
                <DataBox label="input.json（score=0）" color="#6B7280" items={[
                  "5 キープシャイニング: 0点",
                  "8 ジュディーイメル: 0点",
                  "10 テーオーパーセル: 0点",
                  "（全馬スコア空白）",
                ]} />
                <div style={{ textAlign: "center", color: "#374151", fontSize: 18 }}>→</div>
                <DataBox label="Claude 評価（Phase 2）" color="#8B5CF6" items={[
                  "実力50 + 騎手20 + 適性15",
                  "+ 調子10 + 他5 = 100点満点",
                  "枠順補正 ±5点",
                  "展開補正",
                  "本命◎ 1頭宣言",
                ]} />
                <div style={{ textAlign: "center", color: "#374151", fontSize: 18 }}>→</div>
                <DataBox label="input.json（score入り）" color="#10B981" items={[
                  "5 キープシャイニング: 78点",
                  "8 ジュディーイメル: 72点",
                  "10 テーオーパーセル: 62点",
                  "（全馬スコア確定）",
                ]} />
              </div>
              <div style={{
                marginTop: 10, padding: "6px 10px", borderRadius: 4,
                background: "#FBBF2410", border: "1px solid #FBBF2425",
                fontSize: 10, color: "#FBBF24",
              }}>
                💡 課題: Claudeの評価をJSONに書き戻す手順が曖昧。手動コピペ or Claudeに更新JSONを出力させる
              </div>
            </div>

            {/* Calculator */}
            <div style={{
              background: "#111827", borderRadius: 10, padding: 16,
              border: "1px solid #EF444440",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontSize: 13, fontWeight: 700, color: "#F9FAFB" }}>
                  ③ ev_calculator で期待値算出
                </div>
                <Status type="broken" />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 40px 1fr 40px 1fr", alignItems: "center", gap: 0 }}>
                <DataBox label="input.json（完成版）" color="#10B981" items={[
                  "horses[16] + score",
                  "combo_odds.quinella[120]",
                  "combo_odds.wide[120]",
                  "combo_odds.trio[560]",
                ]} />
                <div style={{ textAlign: "center", fontSize: 18 }}>
                  <span style={{ color: "#EF4444" }}>✕</span>
                </div>
                <DataBox label="ev_calculator v2（未実装）" color="#EF4444" items={[
                  "❌ JSON読込機能がない",
                  "❌ combo_oddsを使えない",
                  "❌ 手入力のみ",
                  "→ v2で解消する",
                ]} />
                <div style={{ textAlign: "center", color: "#374151", fontSize: 18 }}>→</div>
                <DataBox label="期待値ランキング" color="#FBBF24" items={[
                  "単勝 EV ランク",
                  "複勝 EV ランク",
                  "馬連 EV TOP10（実オッズ）",
                  "ワイド EV TOP10（実オッズ）",
                  "3連複 EV TOP10（実オッズ）",
                ]} />
              </div>
            </div>

            {/* Summary */}
            <div style={{
              background: "#0D3B2E15", borderRadius: 10, padding: 16,
              border: "1px solid #16653440",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#4ADE80", marginBottom: 8 }}>
                解消後のデータフロー（ev_calculator v2）
              </div>
              <div style={{
                display: "flex", alignItems: "center", gap: 8,
                flexWrap: "wrap", fontSize: 11, color: "#D1D5DB",
              }}>
                <span style={{ padding: "3px 8px", background: "#EF444420", borderRadius: 4, border: "1px solid #EF444430" }}>JRA</span>
                <span style={{ color: "#374151" }}>→</span>
                <span style={{ padding: "3px 8px", background: "#3B82F620", borderRadius: 4, border: "1px solid #3B82F630" }}>scraper v1.4</span>
                <span style={{ color: "#374151" }}>→</span>
                <span style={{ padding: "3px 8px", background: "#10B98120", borderRadius: 4, border: "1px solid #10B98130", fontWeight: 700 }}>input.json</span>
                <span style={{ color: "#374151" }}>→</span>
                <span style={{ padding: "3px 8px", background: "#8B5CF620", borderRadius: 4, border: "1px solid #8B5CF630" }}>Claude評価</span>
                <span style={{ color: "#374151" }}>→</span>
                <span style={{ padding: "3px 8px", background: "#10B98120", borderRadius: 4, border: "1px solid #10B98130", fontWeight: 700 }}>input.json + score</span>
                <span style={{ color: "#374151" }}>→</span>
                <span style={{ padding: "3px 8px", background: "#FBBF2420", borderRadius: 4, border: "1px solid #FBBF2430" }}>ev_calc v2</span>
                <span style={{ color: "#374151" }}>→</span>
                <span style={{ padding: "3px 8px", background: "#4ADE8020", borderRadius: 4, border: "1px solid #4ADE8030", fontWeight: 700 }}>買い目</span>
              </div>
              <div style={{ fontSize: 10, color: "#6B7280", marginTop: 8 }}>
                input.json が全工程の共通データ形式。スクレイパー→Claude→ツールを一本で繋ぐ。
              </div>
            </div>
          </div>
        )}

        {/* =================== ファイル構成 =================== */}
        {view === "ファイル構成" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            <div style={{
              background: "#111827", borderRadius: 10, padding: 16,
              border: "1px solid #1F2937",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#F9FAFB", marginBottom: 12 }}>
                keiba-project/ リポジトリ
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace" }}>
                <FileItem name="docs/" desc="" indent={0} />
                <FileItem name="project_v3_draft.md" desc="指示書v3.0" status="ok" indent={1} />
                <FileItem name="architecture.jsx" desc="アーキテクチャ図(旧)" status="ok" indent={1} />
                <FileItem name="issues_map.jsx" desc="問題点マップ(旧)" status="ok" indent={1} />

                <FileItem name="tools/" desc="" indent={0} />
                <FileItem name="jra_scraper.py" desc="v1.4 → v1.4.1（馬単削除）" status="ok" indent={1} />
                <FileItem name="jra_debug.py" desc="HTMLデバッグ v1" status="ok" indent={1} />
                <FileItem name="jra_debug2.py" desc="HTMLデバッグ v2" status="ok" indent={1} />
                <FileItem name="ev_calculator.jsx" desc="v0.1 → v2（JSON読込+実オッズ）" status="broken" indent={1} />

                <FileItem name="data/" desc="" indent={0} />
                <FileItem name="templates/" desc="" indent={1} />
                <FileItem name="input.json" desc="入力テンプレート" status="ok" indent={2} />
                <FileItem name="result.json" desc="結果テンプレート" status="ok" indent={2} />
                <FileItem name="races/" desc="" indent={1} />
                <FileItem name="20260214_kyoto_4r/" desc="京都4R（検証済）" status="ok" indent={2} />
                <FileItem name="YYYYMMDD_会場_レース名/" desc="今後のレース" indent={2} />
              </div>
            </div>

            {/* What to build */}
            <div style={{
              background: "#111827", borderRadius: 10, padding: 16,
              border: "1px solid #3B82F640",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#60A5FA", marginBottom: 12 }}>
                作るもの（優先順）
              </div>

              {[
                {
                  pri: "P1",
                  color: "#EF4444",
                  title: "ev_calculator.jsx v2",
                  desc: "JSON読込 + 実オッズベース期待値計算",
                  detail: "input.jsonをD&Dで読込。combo_oddsの実オッズで馬連・ワイド・3連複の期待値を正確に算出。評価点の直接編集。",
                },
                {
                  pri: "P2",
                  color: "#F97316",
                  title: "jra_scraper.py v1.4.1",
                  desc: "馬単削除（指示書スコープ外）",
                  detail: "exactaパーサーとタブクリックを削除。JSON構造からもexacta除去。",
                },
                {
                  pri: "P3",
                  color: "#FBBF24",
                  title: "Claude → JSON 書き戻し手順",
                  desc: "評価点をinput.jsonに反映する統一手順",
                  detail: "案A: Claudeが更新済みJSONを出力、案B: ツール上で評価点を直接入力",
                },
              ].map((item, i) => (
                <div key={i} style={{
                  display: "flex", gap: 10, marginBottom: 10, alignItems: "flex-start",
                }}>
                  <span style={{
                    fontSize: 9, fontWeight: 800, color: item.color,
                    background: item.color + "15", border: `1px solid ${item.color}40`,
                    borderRadius: 4, padding: "3px 8px", flexShrink: 0,
                  }}>{item.pri}</span>
                  <div>
                    <div style={{ fontSize: 12, fontWeight: 700, color: "#F9FAFB" }}>{item.title}</div>
                    <div style={{ fontSize: 11, color: "#9CA3AF", marginTop: 1 }}>{item.desc}</div>
                    <div style={{ fontSize: 10, color: "#6B7280", marginTop: 3, lineHeight: 1.5 }}>{item.detail}</div>
                  </div>
                </div>
              ))}
            </div>

            {/* input.json schema */}
            <div style={{
              background: "#111827", borderRadius: 10, padding: 16,
              border: "1px solid #10B98140",
            }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: "#4ADE80", marginBottom: 12 }}>
                input.json = 全工程の共通フォーマット
              </div>
              <pre style={{
                fontSize: 10, color: "#D1D5DB", lineHeight: 1.6,
                background: "#0D1117", padding: 12, borderRadius: 6,
                border: "1px solid #1F2937", overflow: "auto",
                fontFamily: "'JetBrains Mono', monospace",
              }}>{`{
  "race": {
    "date", "venue", "race_number", "name",
    "grade", "surface", "distance", "direction",
    "entries", "weather", "track_condition"
  },
  "parameters": {
    "temperature": 10,    // Claude推奨 → 人間調整
    "budget": 1500,       // グレード別デフォルト
    "top_n": 6
  },
  "horses": [{
    "num", "name",
    "score": 78,          // ← Claudeが埋める
    "score_breakdown": {   // ← Claudeが埋める
      "ability", "jockey", "fitness", "form", "other"
    },
    "odds_win", "odds_place",
    "jockey", "sex_age", "weight", "load_weight",
    "note"                // ← Claudeが埋める
  }],
  "combo_odds": {
    "quinella": [{"combo": [4,10], "odds": 12.3}, ...],  // 120組
    "wide":     [{"combo": [4,10], "odds":  4.5}, ...],  // 120組
    "trio":     [{"combo": [4,7,10], "odds": 85.2}, ...]  // 560組
  }
}`}</pre>
              <div style={{ fontSize: 10, color: "#6B7280", marginTop: 8, lineHeight: 1.6 }}>
                スクレイパーが生成 → Claudeがscore/note記入 → ev_calculatorが読込
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
