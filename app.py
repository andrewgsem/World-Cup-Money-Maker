"""
app.py -- view-only website for the World Cup 2026 predictions.

Pages:
  1. World Cup 2026   -- home: today's games, pulse, groups, finished fixtures
  2. Predictions      -- the model's locked predictions per matchday
  3. Scorecard        -- how good the predictions were

Run locally:   streamlit run app.py
This app ONLY READS files written by your notebook. It never fits or edits.
"""

import os
import pandas as pd
import streamlit as st

from predictions_io import load_log, compute_scorecard

st.set_page_config(page_title="World Cup 2026", page_icon="⚽", layout="wide")

# ----------------------------------------------------------------------------
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

FLAGS = {
    "Mexico": "🇲🇽", "South Africa": "🇿🇦", "South Korea": "🇰🇷", "Czech Republic": "🇨🇿",
    "Canada": "🇨🇦", "Bosnia and Herzegovina": "🇧🇦", "Qatar": "🇶🇦", "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷", "Morocco": "🇲🇦", "Haiti": "🇭🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸", "Paraguay": "🇵🇾", "Australia": "🇦🇺", "Turkey": "🇹🇷",
    "Germany": "🇩🇪", "Curaçao": "🇨🇼", "Ivory Coast": "🇨🇮", "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱", "Japan": "🇯🇵", "Sweden": "🇸🇪", "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪", "Egypt": "🇪🇬", "Iran": "🇮🇷", "New Zealand": "🇳🇿",
    "Spain": "🇪🇸", "Cape Verde": "🇨🇻", "Saudi Arabia": "🇸🇦", "Uruguay": "🇺🇾",
    "France": "🇫🇷", "Senegal": "🇸🇳", "Iraq": "🇮🇶", "Norway": "🇳🇴",
    "Argentina": "🇦🇷", "Algeria": "🇩🇿", "Austria": "🇦🇹", "Jordan": "🇯🇴",
    "Portugal": "🇵🇹", "DR Congo": "🇨🇩", "Uzbekistan": "🇺🇿", "Colombia": "🇨🇴",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Croatia": "🇭🇷", "Ghana": "🇬🇭", "Panama": "🇵🇦",
}


def flag(team: str) -> str:
    return f"{FLAGS.get(team, '')} {team}".strip()


# ----------------------------------------------------------------------------
PLAYED_LIVE = "data_cache/played_live.parquet"
FUTURE_LIVE = "data_cache/future_live.parquet"


@st.cache_data(ttl=300)
def load_wc_results():
    if not os.path.exists(PLAYED_LIVE):
        return pd.DataFrame(columns=["date", "home_team", "away_team",
                                     "home_score", "away_score"])
    df = pd.read_parquet(PLAYED_LIVE)
    df["date"] = pd.to_datetime(df["date"])
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]
    return wc.sort_values("date")


@st.cache_data(ttl=300)
def load_future_fixtures():
    if not os.path.exists(FUTURE_LIVE):
        return pd.DataFrame(columns=["date", "home_team", "away_team"])
    df = pd.read_parquet(FUTURE_LIVE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def standings(group_teams, wc_results):
    rows = {t: dict(P=0, W=0, D=0, L=0, GF=0, GA=0) for t in group_teams}
    g = wc_results[wc_results["home_team"].isin(group_teams)
                   & wc_results["away_team"].isin(group_teams)]
    for m in g.itertuples(index=False):
        hs, as_ = int(m.home_score), int(m.away_score)
        h, a = rows[m.home_team], rows[m.away_team]
        h["P"] += 1; a["P"] += 1
        h["GF"] += hs; h["GA"] += as_
        a["GF"] += as_; a["GA"] += hs
        if hs > as_:
            h["W"] += 1; a["L"] += 1
        elif hs < as_:
            a["W"] += 1; h["L"] += 1
        else:
            h["D"] += 1; a["D"] += 1
    t = pd.DataFrame(rows).T
    t["GD"] = t["GF"] - t["GA"]
    t["Pts"] = 3 * t["W"] + t["D"]
    t = t.sort_values(["Pts", "GD", "GF"], ascending=False)
    t.insert(0, "Team", [flag(x) for x in t.index])
    t["GD"] = t["GD"].map(lambda v: f"+{v}" if v > 0 else str(v))
    t["Goals"] = t["GF"].astype(str) + ":" + t["GA"].astype(str)
    return t.reset_index(drop=True)[["Team", "P", "W", "D", "L", "Goals", "GD", "Pts"]]


def style_standings(df):
    """Qualification coloring: rows 1-2 green (advance), row 3 amber
    (possible best-third), row 4 neutral."""
    def row_style(row):
        if row.name <= 1:
            return ["background-color: rgba(18,110,77,.16)"] * len(row)
        if row.name == 2:
            return ["background-color: rgba(230,170,30,.16)"] * len(row)
        return [""] * len(row)
    return df.style.apply(row_style, axis=1)


def scoreboard_card(home, away, hs, as_, delay=0.0):
    return (
        f"<div class='scoreboard wc-fade' style='animation-delay:{delay:.2f}s;'>"
        f"  <span class='sb-team sb-home'>{flag(home)}</span>"
        f"  <span class='sb-score'>{hs}<span class='sb-dash'>–</span>{as_}</span>"
        f"  <span class='sb-team sb-away'>{flag(away)}</span>"
        f"</div>"
    )


def prob_bar(ph, pdr, pa):
    return (
        "<div class='pbar'>"
        f"<div class='pbar-h' style='width:{ph*100:.0f}%'></div>"
        f"<div class='pbar-d' style='width:{pdr*100:.0f}%'></div>"
        f"<div class='pbar-a' style='width:{pa*100:.0f}%'></div>"
        "</div>"
        f"<div class='pbar-lbl'><span>{ph*100:.0f}%</span>"
        f"<span>draw {pdr*100:.0f}%</span><span>{pa*100:.0f}%</span></div>"
    )


# ----------------------------------------------------------------------------
st.sidebar.title("World Cup 2026")
page = st.sidebar.radio("Page", ["World Cup 2026", "Predictions", "Scorecard"])


# ---- CSS shared by all pages (cards, banners, bars) ----
st.markdown("""
<style>
@keyframes heroIn { from { opacity:0; transform: translateY(16px);} to { opacity:1; transform: translateY(0);} }
.wc-fade { animation: heroIn .8s ease both; }

/* ambient drifting background on every page */
@keyframes ambientDrift {
  0%   { background-position: 0% 0%, 100% 100%; }
  50%  { background-position: 100% 30%, 0% 70%; }
  100% { background-position: 0% 0%, 100% 100%; }
}
[data-testid="stAppViewContainer"] {
  background-image:
    radial-gradient(900px 500px at 15% 10%, rgba(18,110,77,.10), transparent 60%),
    radial-gradient(900px 500px at 85% 90%, rgba(10,35,66,.10), transparent 60%);
  background-size: 200% 200%, 200% 200%;
  animation: ambientDrift 30s ease-in-out infinite;
}

/* stadium-style matchday banner */
.daybanner {
  display:flex; justify-content:space-between; align-items:center;
  background: linear-gradient(90deg, #050d1f, #0a2342 60%, #0c3d2e);
  color:#eaf2ff; border-radius:10px; padding:10px 16px; margin:18px 0 12px 0;
  box-shadow: 0 8px 22px rgba(5,13,31,.25);
}
.daybanner .dayname { font-weight:700; letter-spacing:1px; }
.daybanner .daycount { font-size:12px; opacity:.75; letter-spacing:2px; text-transform:uppercase; }

/* matchup card */
.matchup {
  background: rgba(255,255,255,.6); backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,.7); border-radius: 14px;
  padding: 14px 16px 12px 16px; margin-bottom: 12px;
  box-shadow: 0 8px 28px rgba(10,35,66,.10);
}
.matchup.confident { border-left: 4px solid #126e4d; }
.matchup.tossup    { border-left: 4px solid #e0a92e; }
.matchup .teams { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
.matchup .tname { font-size:16px; }
.matchup .fav   { font-weight:800; text-shadow: 0 0 14px rgba(18,110,77,.55); }
.matchup .vs    { font-size:12px; opacity:.6; letter-spacing:2px; }
.matchup .vs.close { color:#b07d10; opacity:.95; font-weight:700; }
.matchup .headline { text-align:center; font-size:14px; margin:6px 0 8px 0; }
.matchup .headline b { font-size:16px; }
.matchup .smallprint { text-align:center; font-size:11px; opacity:.65; margin-top:6px; }

/* probability bar */
.pbar { display:flex; height:8px; border-radius:6px; overflow:hidden; }
.pbar-h { background:#126e4d; } .pbar-d { background:#b9c2cc; } .pbar-a { background:#b22234; }
.pbar-lbl { display:flex; justify-content:space-between; font-size:11px; opacity:.75; margin-top:4px; }

@media (prefers-reduced-motion: reduce) { .wc-fade { animation:none !important; } }
</style>
""", unsafe_allow_html=True)

log = load_log()
wc = load_wc_results()
future = load_future_fixtures()

# ============================================================== HOME =========
if page == "World Cup 2026":
    # ---- tournament pulse numbers ----
    n_played = len(wc)
    n_goals = int(wc["home_score"].sum() + wc["away_score"].sum()) if n_played else 0
    gpg = f"{n_goals / n_played:.2f}" if n_played else "–"

    st.markdown(
        f"""
        <style>
        /* ---------- glass panels: dataframes sit on frosted glass ---------- */
        [data-testid="stDataFrame"] {{
          background: rgba(255,255,255,.55);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255,255,255,.65);
          border-radius: 14px;
          padding: 8px;
          box-shadow: 0 8px 28px rgba(10,35,66,.10);
        }}

        /* ---------- hero: animated stadium night ---------- */
        .stadium {{
          position: relative; overflow: hidden; border-radius: 16px;
          height: 340px; color: white; text-align: center;
          background: linear-gradient(180deg, #050d1f 0%, #0a2342 55%, #0c3d2e 100%);
          box-shadow: 0 18px 50px rgba(5,13,31,.35);
        }}
        @keyframes aurora1 {{ 0%,100% {{ transform: translate(-12%, -8%) scale(1); }}
                             50%     {{ transform: translate(14%, 6%)  scale(1.25); }} }}
        @keyframes aurora2 {{ 0%,100% {{ transform: translate(10%, 4%)  scale(1.1); }}
                             50%     {{ transform: translate(-14%, -6%) scale(.9); }} }}
        .stadium .aurora {{ position: absolute; inset: -30%; filter: blur(46px); opacity: .55; }}
        .stadium .aurora.one {{
          background: radial-gradient(40% 32% at 30% 28%, rgba(178,34,52,.55), transparent 70%),
                      radial-gradient(36% 30% at 72% 22%, rgba(18,110,77,.55), transparent 70%);
          animation: aurora1 16s ease-in-out infinite;
        }}
        .stadium .aurora.two {{
          background: radial-gradient(34% 30% at 56% 36%, rgba(64,120,200,.45), transparent 70%);
          animation: aurora2 21s ease-in-out infinite;
        }}
        @keyframes sweepL {{ 0%,100% {{ transform: rotate(16deg); }} 50% {{ transform: rotate(28deg); }} }}
        @keyframes sweepR {{ 0%,100% {{ transform: rotate(-16deg); }} 50% {{ transform: rotate(-28deg); }} }}
        .stadium .beam {{
          position: absolute; top: -30px; width: 130px; height: 440px; opacity: .16;
          background: linear-gradient(to bottom, rgba(255,255,240,.9), transparent 75%);
          clip-path: polygon(42% 0, 58% 0, 100% 100%, 0 100%);
        }}
        .stadium .beam.left  {{ left: 6%;  transform-origin: top center; animation: sweepL 9s ease-in-out infinite; }}
        .stadium .beam.right {{ right: 6%; transform-origin: top center; animation: sweepR 11s ease-in-out infinite; }}
        .stadium .pitch {{
          position: absolute; bottom: -8px; left: -12%; right: -12%; height: 120px;
          background: repeating-linear-gradient(90deg, #0e4a36 0 64px, #11583f 64px 128px);
          transform: perspective(340px) rotateX(58deg);
          border-top: 2px solid rgba(255,255,255,.25);
        }}
        @keyframes lineGlow {{ 0%,100% {{ opacity:.35; }} 50% {{ opacity:.85; }} }}
        .stadium .midline {{
          position: absolute; bottom: 0; left: 50%; width: 2px; height: 118px;
          background: rgba(255,255,255,.8); transform: translateX(-50%);
          animation: lineGlow 4s ease-in-out infinite;
        }}
        @keyframes rise {{
          0%   {{ transform: translateY(110%) scale(.6); opacity: 0; }}
          12%  {{ opacity: .9; }}
          100% {{ transform: translateY(-30%) scale(1.1); opacity: 0; }}
        }}
        .stadium .p {{ position: absolute; bottom: 0; border-radius: 50%;
                      background: rgba(255,255,255,.85); }}
        .stadium .p1 {{ left: 12%; width: 3px; height: 3px; animation: rise 9s  linear infinite; }}
        .stadium .p2 {{ left: 26%; width: 2px; height: 2px; animation: rise 13s linear infinite 2s; }}
        .stadium .p3 {{ left: 47%; width: 3px; height: 3px; animation: rise 11s linear infinite 4s; }}
        .stadium .p4 {{ left: 64%; width: 2px; height: 2px; animation: rise 14s linear infinite 1s; }}
        .stadium .p5 {{ left: 81%; width: 3px; height: 3px; animation: rise 10s linear infinite 5s; }}
        .stadium .p6 {{ left: 91%; width: 2px; height: 2px; animation: rise 12s linear infinite 3s; }}

        /* ---------- film grain + vignette over the hero ---------- */
        @keyframes grainShift {{
          0% {{ transform: translate(0,0); }} 25% {{ transform: translate(-4%,3%); }}
          50% {{ transform: translate(3%,-4%); }} 75% {{ transform: translate(-3%,-2%); }}
          100% {{ transform: translate(0,0); }}
        }}
        .stadium .grain {{
          position: absolute; inset: -20%; z-index: 2; pointer-events: none; opacity: .07;
          background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)'/%3E%3C/svg%3E");
          animation: grainShift 1.6s steps(4) infinite;
        }}
        .stadium .vignette {{
          position: absolute; inset: 0; z-index: 2; pointer-events: none;
          background: radial-gradient(120% 90% at 50% 40%, transparent 55%, rgba(0,0,0,.45) 100%);
        }}

        @keyframes heroIn {{ from {{ opacity: 0; transform: translateY(16px); }}
                            to   {{ opacity: 1; transform: translateY(0); }} }}
        .stadium .content {{ position: relative; z-index: 3; padding-top: 52px;
                            animation: heroIn 1.1s ease both; }}
        .stadium h1 {{ margin: 0; font-size: 44px; letter-spacing: 2px;
                      text-shadow: 0 2px 18px rgba(0,0,0,.55); }}
        .stadium .sub {{ opacity: .85; letter-spacing: 3px; font-size: 13px;
                        text-transform: uppercase; margin-top: 6px; }}
        .stadium .hosts {{ margin-top: 8px; font-size: 24px; letter-spacing: 10px; }}
        .stadium .pulse {{ margin-top: 14px; display: flex; justify-content: center;
                          gap: 34px; font-size: 13px; letter-spacing: 1px; opacity: .9; }}
        .stadium .pulse b {{ display: block; font-size: 22px; letter-spacing: 0; }}

        .wc-fade {{ animation: heroIn .8s ease both; }}

        /* ---------- scoreboard match cards ---------- */
        .scoreboard {{
          display: flex; align-items: center; justify-content: space-between;
          background: linear-gradient(180deg, #0b1c33, #122a47);
          border: 1px solid rgba(120,160,220,.25); border-radius: 12px;
          padding: 12px 16px; color: #eaf2ff; margin-bottom: 10px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 8px 22px rgba(5,13,31,.25);
        }}
        .scoreboard .sb-team {{ flex: 1; font-size: 15px; }}
        .scoreboard .sb-home {{ text-align: left; }}
        .scoreboard .sb-away {{ text-align: right; }}
        .scoreboard .sb-score {{
          font-variant-numeric: tabular-nums; font-weight: 800; font-size: 24px;
          color: #ffd24d; text-shadow: 0 0 12px rgba(255,210,77,.55);
          padding: 0 14px; white-space: nowrap;
        }}
        .scoreboard .sb-dash {{ opacity: .6; padding: 0 6px; font-weight: 400; }}

        /* ---------- today strip: fixture card + probability bar ---------- */
        .todaycard {{
          background: rgba(255,255,255,.6); backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          border: 1px solid rgba(255,255,255,.7); border-radius: 14px;
          padding: 14px 16px 10px 16px; margin-bottom: 10px;
          box-shadow: 0 8px 28px rgba(10,35,66,.10);
        }}
        .todaycard .teams {{ display:flex; justify-content: space-between;
                            font-weight: 600; margin-bottom: 8px; }}
        .pbar {{ display: flex; height: 8px; border-radius: 6px; overflow: hidden; }}
        .pbar-h {{ background: #126e4d; }}
        .pbar-d {{ background: #b9c2cc; }}
        .pbar-a {{ background: #b22234; }}
        .pbar-lbl {{ display: flex; justify-content: space-between; font-size: 11px;
                    opacity: .75; margin-top: 4px; }}

        @media (prefers-reduced-motion: reduce) {{
          .stadium *, [data-testid="stAppViewContainer"], .wc-fade {{ animation: none !important; }}
        }}
        </style>

        <div class="stadium">
          <div class="aurora one"></div>
          <div class="aurora two"></div>
          <div class="beam left"></div>
          <div class="beam right"></div>
          <div class="p p1"></div><div class="p p2"></div><div class="p p3"></div>
          <div class="p p4"></div><div class="p p5"></div><div class="p p6"></div>
          <div class="content">
            <h1>FIFA WORLD CUP 2026</h1>
            <div class="sub">Canada · Mexico · United States — 48 teams · 12 groups · one trophy</div>
            <div class="hosts">🇨🇦 🇲🇽 🇺🇸</div>
            <div class="pulse">
              <span><b>{n_played}</b> matches played</span>
              <span><b>{n_goals}</b> goals</span>
              <span><b>{gpg}</b> goals / game</span>
            </div>
          </div>
          <div class="pitch"></div>
          <div class="midline"></div>
          <div class="grain"></div>
          <div class="vignette"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # ---- today's matches ----
    today = pd.Timestamp.now().normalize()
    todays = future[future["date"].dt.normalize() == today]
    if not todays.empty:
        st.subheader("Today's matches")
        # probabilities from the locked prediction log, if published
        lg = log.copy()
        if not lg.empty:
            lg["d"] = pd.to_datetime(lg["date"]).dt.normalize()
        cols = st.columns(min(len(todays), 3))
        for i, m in enumerate(todays.itertuples(index=False)):
            with cols[i % len(cols)]:
                p = None
                if not lg.empty:
                    hit = lg[(lg["d"] == today) & (lg["home"] == m.home_team)
                             & (lg["away"] == m.away_team)]
                    if len(hit):
                        r = hit.iloc[0]
                        p = (r["p_home"], r["p_draw"], r["p_away"])
                inner = (f"<div class='teams'><span>{flag(m.home_team)}</span>"
                         f"<span>vs</span><span>{flag(m.away_team)}</span></div>")
                inner += prob_bar(*p) if p else \
                    "<div class='pbar-lbl'><span>prediction not published yet</span></div>"
                st.markdown(f"<div class='todaycard wc-fade' "
                            f"style='animation-delay:{i*0.1:.2f}s'>{inner}</div>",
                            unsafe_allow_html=True)
        st.write("")

    # ---- finished matches as scoreboards ----
    if not wc.empty:
        st.subheader("Finished matches")
        for day in sorted(wc["date"].dt.date.unique(), reverse=True):
            d = wc[wc["date"].dt.date == day]
            st.markdown(f"**{pd.Timestamp(day).strftime('%A, %d %B')}**")
            cols = st.columns(min(len(d), 3))
            for i, m in enumerate(d.itertuples(index=False)):
                with cols[i % len(cols)]:
                    st.markdown(scoreboard_card(m.home_team, m.away_team,
                                                int(m.home_score), int(m.away_score),
                                                delay=i * 0.12),
                                unsafe_allow_html=True)
        st.write("")

    # ---- group tables (2 per row, qualification-colored) ----
    st.subheader("Groups")
    letters = list(GROUPS.keys())
    for row_start in range(0, 12, 2):
        cols = st.columns(2)
        for j, letter in enumerate(letters[row_start:row_start + 2]):
            with cols[j]:
                st.markdown(f"##### Group {letter}")
                st.dataframe(style_standings(standings(GROUPS[letter], wc)),
                             hide_index=True, use_container_width=True)

    st.caption("🟩 top two advance to the round of 32 · 🟨 third place may qualify "
               "among the eight best thirds.")
    st.info("🥊 Knockout stage bracket will appear here once the group phase ends.")

# ========================================================== PREDICTIONS ======
elif page == "Predictions":
    n_upcoming = int(log["home_score"].isna().sum()) if not log.empty else 0
    st.markdown(
        f"""
        <style>
        .tactics {{
          position: relative; overflow: hidden; border-radius: 16px;
          height: 200px; color: #eaf6ee; text-align: center;
          background:
            linear-gradient(180deg, rgba(8,28,20,.0) 0%, rgba(5,18,13,.55) 100%),
            repeating-linear-gradient(0deg,  transparent 0 39px, rgba(255,255,255,.05) 39px 40px),
            repeating-linear-gradient(90deg, transparent 0 39px, rgba(255,255,255,.05) 39px 40px),
            linear-gradient(135deg, #0b3a2a, #0e4a36 60%, #0a2f3d);
          box-shadow: 0 14px 40px rgba(5,18,13,.35);
        }}
        @keyframes chalkDraw {{ to {{ stroke-dashoffset: 0; }} }}
        @keyframes chalkLoop {{
          0%, 72%  {{ opacity: 1; }}
          86%      {{ opacity: 0; }}
          100%     {{ opacity: 1; }}
        }}
        /* moving aesthetics on the board itself -- colors unchanged */
        @keyframes boardGlow {{ 0%,100% {{ transform: translate(0,0) scale(1); }}
                               50%     {{ transform: translate(5%,-4%) scale(1.08); }} }}
        .tactics .bglow {{ position: absolute; inset: -30%; filter: blur(42px); opacity: .5;
          background: radial-gradient(32% 38% at 28% 35%, rgba(40,160,110,.35), transparent 70%),
                      radial-gradient(30% 36% at 72% 60%, rgba(20,90,130,.30), transparent 70%);
          animation: boardGlow 14s ease-in-out infinite; }}
        @keyframes boardSweep {{ 0% {{ left: -35%; }} 55%, 100% {{ left: 110%; }} }}
        .tactics .sweep {{ position: absolute; top: 0; bottom: 0; width: 26%;
          transform: skewX(-14deg);
          background: linear-gradient(100deg, transparent,
                      rgba(255,255,255,.05) 48%, rgba(255,255,255,.09) 52%, transparent);
          animation: boardSweep 8s ease-in-out infinite; }}
        @keyframes dustFloat {{ 0% {{ transform: translateY(16px); opacity: 0; }}
                               20% {{ opacity: .7; }}
                               100% {{ transform: translateY(-170px); opacity: 0; }} }}
        .tactics .dust {{ position: absolute; bottom: 8px; width: 3px; height: 3px;
                         border-radius: 50%; background: rgba(255,255,255,.7); }}
        .tactics .du1 {{ left: 12%; animation: dustFloat 8s  linear infinite; }}
        .tactics .du2 {{ left: 31%; animation: dustFloat 11s linear infinite 2s; }}
        .tactics .du3 {{ left: 52%; animation: dustFloat 9s  linear infinite 4s; }}
        .tactics .du4 {{ left: 71%; animation: dustFloat 12s linear infinite 1s; }}
        .tactics .du5 {{ left: 88%; animation: dustFloat 10s linear infinite 3s; }}
        .tactics svg {{ position: absolute; inset: 0; width: 100%; height: 100%;
                       opacity: .55; animation: chalkLoop 16s ease-in-out infinite; }}
        .tactics .lines {{ fill: none; stroke: rgba(255,255,255,.22); stroke-width: 1.5; }}
        .tactics .play {{ fill: none; stroke: rgba(255,255,255,.8); stroke-width: 2.5;
                         stroke-linecap: round; stroke-dasharray: 9 8; }}
        .tactics .ahead {{ fill: none; stroke: rgba(255,255,255,.8); stroke-width: 2.5;
                          stroke-linecap: round; opacity: 0; }}
        .tactics .draw1 {{ stroke-dashoffset: 600; animation: chalkDraw 3.2s ease forwards .3s; }}
        .tactics .draw2 {{ stroke-dashoffset: 600; animation: chalkDraw 3.2s ease forwards 1.0s; }}
        .tactics .draw3 {{ stroke-dashoffset: 600; animation: chalkDraw 3.2s ease forwards 1.7s; }}
        .tactics .draw4 {{ stroke-dashoffset: 600; animation: chalkDraw 3.2s ease forwards 2.4s; }}
        .tactics .draw5 {{ stroke-dashoffset: 600; animation: chalkDraw 3.2s ease forwards 3.1s; }}
        @keyframes aheadIn {{ to {{ opacity: 1; }} }}
        .tactics .ahead.draw1 {{ animation: aheadIn .4s ease forwards 3.2s; }}
        .tactics .ahead.draw2 {{ animation: aheadIn .4s ease forwards 3.9s; }}
        .tactics .ahead.draw3 {{ animation: aheadIn .4s ease forwards 4.6s; }}
        .tactics .ahead.draw4 {{ animation: aheadIn .4s ease forwards 5.3s; }}
        .tactics .ahead.draw5 {{ animation: aheadIn .4s ease forwards 6.0s; }}
        .tactics .mark  {{ fill: none; stroke: rgba(255,220,120,.85); stroke-width: 3; }}
        .tactics .xmark {{ fill: none; stroke: rgba(255,220,120,.85); stroke-width: 3;
                          stroke-linecap: round; }}
        .tactics .dot   {{ fill: rgba(140,200,255,.9); }}
        .tactics .content {{ position: relative; z-index: 2; padding-top: 56px;
                            animation: heroIn 1s ease both; }}
        .tactics h1 {{ margin: 0; font-size: 36px; letter-spacing: 3px;
                      text-shadow: 0 2px 14px rgba(0,0,0,.6); }}
        .tactics .sub {{ opacity: .8; letter-spacing: 3px; font-size: 12px;
                        text-transform: uppercase; margin-top: 8px; }}
        @media (prefers-reduced-motion: reduce) {{ .tactics svg, .tactics .play {{ animation: none !important; }} }}
        </style>

        <div class="tactics">
          <div class="bglow"></div>
          <div class="sweep"></div>
          <div class="dust du1"></div><div class="dust du2"></div><div class="dust du3"></div>
          <div class="dust du4"></div><div class="dust du5"></div>
          <svg viewBox="0 0 800 200" preserveAspectRatio="none">
            <!-- pitch markings: the quiet base layer -->
            <g class="lines">
              <rect x="8" y="10" width="784" height="180" rx="4"/>
              <line x1="400" y1="10" x2="400" y2="190"/>
              <circle cx="400" cy="100" r="34"/>
              <rect x="8"   y="55" width="70" height="90"/>
              <rect x="722" y="55" width="70" height="90"/>
              <rect x="8"   y="80" width="28" height="40"/>
              <rect x="764" y="80" width="28" height="40"/>
              <path d="M 78 78 A 30 30 0 0 1 78 122"/>
              <path d="M 722 78 A 30 30 0 0 0 722 122"/>
            </g>
            <!-- the chalked plays: drawn in sequence -->
            <path class="play draw1" d="M 60 160 C 180 120, 240 70, 380 86"/>
            <path class="play draw2" d="M 420 130 C 530 150, 620 60, 740 52"/>
            <path class="play draw3" d="M 120 50 C 220 40, 300 130, 360 150"/>
            <path class="play draw4" d="M 480 60 C 560 90, 600 140, 700 150"/>
            <path class="play draw5" d="M 200 178 C 320 168, 420 60, 540 44"/>
            <!-- arrowheads at the receiving end of each run -->
            <path class="ahead draw1" d="M 380 86 l -14 -7 M 380 86 l -10 12"/>
            <path class="ahead draw2" d="M 740 52 l -15 -3 M 740 52 l -8 13"/>
            <path class="ahead draw3" d="M 360 150 l -15 -6 M 360 150 l -7 -14"/>
            <path class="ahead draw4" d="M 700 150 l -16 -2 M 700 150 l -9 -13"/>
            <path class="ahead draw5" d="M 540 44 l -16 2 M 540 44 l -6 15"/>
            <!-- our side: a back four plus runners (chalk circles) -->
            <circle class="mark" cx="105" cy="40"  r="9"/>
            <circle class="mark" cx="90"  cy="95"  r="9"/>
            <circle class="mark" cx="105" cy="155" r="9"/>
            <circle class="mark" cx="240" cy="70"  r="9"/>
            <circle class="mark" cx="200" cy="178" r="9"/>
            <circle class="mark" cx="620" cy="60"  r="9"/>
            <!-- their side: X marks to attack -->
            <path class="xmark" d="M 372 78  l 16 16 M 388 78  l -16 16"/>
            <path class="xmark" d="M 732 44  l 16 16 M 748 44  l -16 16"/>
            <path class="xmark" d="M 532 36  l 16 16 M 548 36  l -16 16"/>
            <path class="xmark" d="M 352 142 l 16 16 M 368 142 l -16 16"/>
            <path class="xmark" d="M 692 142 l 16 16 M 708 142 l -16 16"/>
            <!-- the ball -->
            <circle class="dot" cx="60"  cy="160" r="6"/>
            <circle class="dot" cx="420" cy="130" r="6"/>
            <circle class="dot" cx="120" cy="50"  r="5"/>
            <circle class="dot" cx="480" cy="60"  r="5"/>
          </svg>
          <div class="content">
            <h1>THE PREDICTIONS</h1>
            <div class="sub">the model's game plan — {n_upcoming} fixtures called, results pending</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if log.empty:
        st.info("No predictions yet. Run the updater to publish the first matchday.")
        st.stop()

    upcoming = log[log["home_score"].isna()].copy()
    if upcoming.empty:
        st.info("No upcoming fixtures predicted right now.")
    else:
        upcoming["date"] = pd.to_datetime(upcoming["date"]).dt.date
        has_oc = "pred_score_outcome" in upcoming.columns
        for day in sorted(upcoming["date"].unique()):
            d = upcoming[upcoming["date"] == day]
            st.markdown(
                f"<div class='daybanner wc-fade'>"
                f"<span class='dayname'>{pd.Timestamp(day).strftime('%A, %d %B %Y')}</span>"
                f"<span class='daycount'>{len(d)} match{'es' if len(d) != 1 else ''}</span></div>",
                unsafe_allow_html=True)
            cols = st.columns(min(len(d), 2))
            for i, r in enumerate(d.itertuples(index=False)):
                ph, pdr, pa = float(r.p_home), float(r.p_draw), float(r.p_away)
                fav_p = max(ph, pa)
                if ph >= pa:
                    fav, other, outcome = r.home, r.away, f"{flag(r.home)} win"
                else:
                    fav, other, outcome = r.away, r.home, f"{flag(r.away)} win"
                if max(ph, pdr, pa) == pdr:
                    outcome = "Draw"
                conf_cls = "confident" if fav_p >= 0.65 else ("tossup" if fav_p <= 0.45 else "")
                vs_cls = "vs close" if conf_cls == "tossup" else "vs"
                vs_txt = "too close to call" if conf_cls == "tossup" else "vs"
                oc = getattr(r, "pred_score_outcome", None) if has_oc else None
                oc = oc if (oc is not None and not pd.isna(oc)) else None
                headline_score = oc if oc else r.pred_score
                home_cls = "tname fav" if fav == r.home and outcome != "Draw" else "tname"
                away_cls = "tname fav" if fav == r.away and outcome != "Draw" else "tname"
                card = (
                    f"<div class='matchup wc-fade {conf_cls}' style='animation-delay:{i*0.08:.2f}s'>"
                    f"<div class='teams'><span class='{home_cls}'>{flag(r.home)}</span>"
                    f"<span class='{vs_cls}'>{vs_txt}</span>"
                    f"<span class='{away_cls}'>{flag(r.away)}</span></div>"
                    f"<div class='pbar'><div class='pbar-h' style='width:{ph*100:.0f}%'></div>"
                    f"<div class='pbar-d' style='width:{pdr*100:.0f}%'></div>"
                    f"<div class='pbar-a' style='width:{pa*100:.0f}%'></div></div>"
                    f"<div class='pbar-lbl'><span>{ph*100:.0f}%</span>"
                    f"<span>draw {pdr*100:.0f}%</span><span>{pa*100:.0f}%</span></div>"
                    f"<div class='headline'>Model lean: <b>{outcome}</b> · most likely <b>{headline_score}</b></div>"
                    f"<div class='smallprint'>most likely exact score overall: {r.pred_score}</div>"
                    f"</div>"
                )
                with cols[i % len(cols)]:
                    st.markdown(card, unsafe_allow_html=True)

# ============================================================ SCORECARD ======
else:
    played, counts = compute_scorecard(log)
    total = counts["total"]
    cs, co, cn = counts["correct_score"], counts["correct_outcome"], counts["nothing"]
    hitrate = (cs + co) / total * 100 if total else 0

    st.markdown(
        f"""
        <style>
        .verdict {{
          position: relative; overflow: hidden; border-radius: 16px;
          height: 190px; color: #f5f0e6; text-align: center;
          background: linear-gradient(160deg, #0a1626, #10243f 55%, #0d1b2e);
          box-shadow: 0 14px 40px rgba(5,13,31,.35);
        }}
        @keyframes vGlow {{ 0%,100% {{ transform: translate(0,0); }} 50% {{ transform: translate(6%,-6%); }} }}
        .verdict .glow {{ position: absolute; inset: -40%; filter: blur(44px); opacity: .8;
          background: radial-gradient(30% 40% at 30% 30%, rgba(255,210,77,.16), transparent 70%),
                      radial-gradient(30% 40% at 70% 62%, rgba(64,120,200,.20), transparent 70%);
          animation: vGlow 18s ease-in-out infinite; }}
        @keyframes vSweep {{ 0% {{ left: -35%; }} 60%, 100% {{ left: 110%; }} }}
        .verdict .sweep {{ position: absolute; top: 0; bottom: 0; width: 28%;
          transform: skewX(-12deg);
          background: linear-gradient(100deg, transparent, rgba(255,255,255,.04) 45%,
                      rgba(255,215,80,.10) 50%, rgba(255,255,255,.04) 55%, transparent);
          animation: vSweep 7.5s ease-in-out infinite; }}
        .verdict .content {{ position: relative; z-index: 2; padding-top: 50px;
                            animation: heroIn 1s ease both; }}
        .verdict h1 {{ margin: 0; font-size: 36px; letter-spacing: 3px;
                      text-shadow: 0 2px 14px rgba(0,0,0,.6); }}
        @keyframes ruleGrow {{ from {{ width: 0; opacity: 0; }} }}
        .verdict .rule {{ width: 140px; height: 3px; margin: 12px auto 10px auto;
          background: linear-gradient(90deg, transparent, #ffd24d, transparent);
          animation: ruleGrow 1.2s ease both .3s; }}
        .verdict .sub {{ opacity: .8; letter-spacing: 3px; font-size: 12px; text-transform: uppercase; }}

        .vt-grid {{ display: flex; gap: 14px; margin-top: 18px; }}
        .vt {{ flex: 1; text-align: center; border-radius: 14px; padding: 18px 14px 14px 14px;
              background: rgba(255,255,255,.6); backdrop-filter: blur(10px);
              -webkit-backdrop-filter: blur(10px);
              border: 1px solid rgba(255,255,255,.7); border-top: 4px solid var(--c);
              box-shadow: 0 8px 28px rgba(10,35,66,.10); }}
        .vt .big {{ font-size: 42px; font-weight: 800; line-height: 1; color: var(--c); }}
        .vt .lbl {{ font-size: 11px; letter-spacing: 2px; text-transform: uppercase;
                   opacity: .65; margin-top: 6px; }}
        .vt .pct {{ font-size: 13px; opacity: .75; margin-top: 2px; }}
        @keyframes growX {{ from {{ transform: scaleX(0); }} }}
        .vt .bar {{ height: 8px; border-radius: 6px; overflow: hidden;
                   background: rgba(10,35,66,.08); margin-top: 12px; }}
        .vt .bar i {{ display: block; height: 100%; background: var(--c);
                     transform-origin: left; animation: growX 1.1s ease both .2s; }}

        .mrow {{ display: flex; align-items: center; justify-content: space-between;
                gap: 14px; border-radius: 12px; padding: 10px 14px; margin-bottom: 8px;
                background: rgba(255,255,255,.6); backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,.7);
                border-left: 4px solid var(--c);
                box-shadow: 0 6px 20px rgba(10,35,66,.08); }}
        .mrow .when  {{ font-size: 11px; opacity: .6; min-width: 64px; }}
        .mrow .fixt  {{ flex: 1; font-weight: 600; }}
        .mrow .mini  {{ font-variant-numeric: tabular-nums; font-weight: 800;
                       border-radius: 8px; padding: 3px 12px; white-space: nowrap; }}
        .mrow .ghost {{ color: #5a708c; border: 1.5px dashed #9fb0c4;
                       background: rgba(255,255,255,.4); }}
        .mrow .real  {{ background: linear-gradient(180deg, #0b1c33, #122a47);
                       color: #ffd24d; text-shadow: 0 0 10px rgba(255,210,77,.5); }}
        .mrow .arrow {{ opacity: .45; font-size: 12px; }}
        .mrow .chip  {{ font-size: 12px; font-weight: 700; border-radius: 999px;
                       padding: 4px 12px; white-space: nowrap;
                       color: var(--c); background: color-mix(in srgb, var(--c) 14%, white); }}
        </style>

        <div class="verdict">
          <div class="glow"></div>
          <div class="sweep"></div>
          <div class="content">
            <h1>THE SCORECARD</h1>
            <div class="rule"></div>
            <div class="sub">{total} matches judged · outcome hit rate {hitrate:.0f}%</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if total == 0:
        st.write("")
        st.info("No results recorded yet — the scorecard fills in as matches finish.")
        st.stop()

    # ---- the three verdict tiles ----
    def tile(color, n, label, pct):
        return (f"<div class='vt wc-fade' style='--c:{color}'>"
                f"<div class='big'>{n}</div><div class='lbl'>{label}</div>"
                f"<div class='pct'>{pct:.0f}% of {total}</div>"
                f"<div class='bar'><i style='transform:scaleX({pct/100:.3f})'></i></div></div>")

    st.markdown(
        "<div class='vt-grid'>"
        + tile("#126e4d", cs, "correct score", cs / total * 100)
        + tile("#e0a92e", co, "correct outcome", co / total * 100)
        + tile("#b22234", cn, "nothing correct", cn / total * 100)
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption("Correct score = exact scoreline predicted. Correct outcome = right "
               "winner/draw, wrong score. Nothing = wrong outcome.")

    # ---- match by match ----
    st.subheader("Match by match")
    cfg = {"correct_score":   ("#126e4d", "✅ exact score"),
           "correct_outcome": ("#e0a92e", "🟡 outcome"),
           "nothing":         ("#b22234", "❌ miss")}
    det = played.copy()
    det["date"] = pd.to_datetime(det["date"])
    det = det.sort_values("date", ascending=False)
    for i, r in enumerate(det.itertuples(index=False)):
        color, chip = cfg[r.bucket]
        actual = f"{int(r.home_score)}-{int(r.away_score)}"
        row = (f"<div class='mrow wc-fade' style='--c:{color}; animation-delay:{i*0.06:.2f}s'>"
               f"<span class='when'>{pd.Timestamp(r.date).strftime('%d %b')}</span>"
               f"<span class='fixt'>{flag(r.home)} <span style='opacity:.5'>vs</span> {flag(r.away)}</span>"
               f"<span class='mini ghost'>{r.eval_score}</span>"
               f"<span class='arrow'>➜</span>"
               f"<span class='mini real'>{actual}</span>"
               f"<span class='chip'>{chip}</span>"
               f"</div>")
        st.markdown(row, unsafe_allow_html=True)