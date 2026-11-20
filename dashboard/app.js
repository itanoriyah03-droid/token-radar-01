// Token Radar's plain dashboard. It reads window.RADAR (written beside dashboard/ by main.ts)
// and renders the ranked board, a filterable table, a per-protocol detail view with the model's
// own terms and the protocol's farm plan, and the live-token set. No network: every byte it
// shows is in data.js. Query parameters preset the view, so a link can open one protocol's
// detail (?slug=...) or one filtered board (?verdict=FARM, ?category=Lending, ?list=watchlist).
(function () {
  'use strict';

  var DATA = window.RADAR;
  var main = document.getElementById('board');
  if (!DATA) { main.innerHTML = '<p class="empty">No radar data loaded.</p>'; main.setAttribute('aria-busy', 'false'); return; }
  var params = new URLSearchParams(window.location.search);
  var state = {
    q: params.get('q') || '',
    category: params.get('category') || 'all',
    verdict: params.get('verdict') || 'all',
    list: params.get('list') || 'all',
    sort: params.get('sort') || 'implied',
    dir: -1,
    tab: 'board',
    selected: params.get('slug') || null
  };
  var nf = new Intl.NumberFormat('en-US');
  function fmtUsd(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    if (Math.abs(n) >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (Math.abs(n) >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (Math.abs(n) >= 1e3) return '$' + Math.round(n / 1e3) + 'K';
    return '$' + Math.round(n);
  }
  function fmtFull(n) { return (n === null || n === undefined || !isFinite(n)) ? 'n/a' : '$' + nf.format(Math.round(n)); }
  function fmtPct(n) {
    if (n === null || n === undefined || !isFinite(n)) return 'n/a';
    return (n > 0 ? '+' : '') + Number(n).toFixed(2) + '%';
  }
  function fmtAge(days) {
    if (days === null || days === undefined) return 'unknown';
    if (days >= 365) return (days / 365).toFixed(1) + 'y';
    return Math.max(0, Math.round(days)) + 'd';
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
  function bySlug(slug) { for (var i = 0; i < DATA.rows.length; i++) if (DATA.rows[i].slug === slug) return DATA.rows[i]; return null; }

  // ---- header, KPIs, reconciliation, model box
  document.getElementById('asof').textContent = DATA.as_of;
  document.getElementById('floor').textContent = fmtUsd(DATA.floor_usd);
  document.getElementById('mver').textContent = DATA.model_version;
  var v = DATA.validation;
  var kpis = [
    [fmtUsd(DATA.total_implied_usd), 'total implied value, ' + DATA.board_size + ' rows'],
    [String(DATA.board_size), 'rows: ' + DATA.watchlist_size + ' watchlist + ' + DATA.early_size + ' early'],
    [String(DATA.verdict_counts.FARM), 'FARM verdicts beat $25/hr after costs'],
    ['x' + Number(v.loo_median_miss).toFixed(2), 'leave-one-out median miss, fitted multiple'],
    [Number(v.odds.cv_brier).toFixed(3), 'token-odds Brier, 5-fold CV (base ' + Number(v.odds.base_brier).toFixed(3) + ')'],
    [(Number(v.coverage_p25_p75) * 100).toFixed(0) + '% / ' + (Number(v.coverage_p10_p90) * 100).toFixed(0) + '%', 'actual held-out coverage, P25–P75 / P10–P90']
  ];
  document.getElementById('kpis').innerHTML = kpis.map(function (k) { return '<div class="kpi"><b>' + esc(k[0]) + '</b><span>' + esc(k[1]) + '</span></div>'; }).join('');
  var total = 0;
  DATA.reconciliation.forEach(function (g) { total += g.count; });
  document.getElementById('recon').innerHTML = '<span class="pill"><b>' + total + ' protocols reconciled</b> <small>every protocol in exactly one group</small></span>' +
    DATA.reconciliation.map(function (g) { return '<span class="pill"><b>' + g.count + '</b> ' + esc(g.group) + ' <small>' + esc(g.note) + '</small></span>'; }).join('');
  document.getElementById('modelbox').innerHTML =
    '<h2>How the value is fitted — and how it was tested</h2>' +
    '<p>Multiple: <code>log(mcap/TVL) ~ 1 + (log10 TVL − 8) + (log10 TVL − 8)² + (log1p age − 5.5) + clamped 7d growth + category</code>, ' +
    'ridge λ=' + DATA.model.ridge_lambda + ', Huber-robust, fitted on ' + v.n_train + ' protocols whose token trades with a positive cap ' +
    '(' + v.dropped_zero_mcap + ' more record a token with a zero cap and cannot enter a log fit). ' +
    'Coefficients: intercept ' + v.coefficients.intercept + ', log TVL ' + v.coefficients.log_tvl + ', log TVL² ' + v.coefficients.log_tvl_sq +
    ', log age ' + v.coefficients.log_age + ', growth ' + v.coefficients.growth + '. A hot week barely moves the multiple — the fit says so, and the board keeps the fitted term rather than a wished-for one.</p>' +
    '<p>Held out, twice: leave-one-out median miss <b>x' + Number(v.loo_median_miss).toFixed(2) + '</b> (mean x' + Number(v.loo_mean_miss).toFixed(2) + '); ' +
    'walk-forward by listed date <b>x' + v.walk_forward.map(function (w) { return Number(w.median_miss).toFixed(2); }).join(' / x') + '</b> (fit the older half, then the older 70%, test the newer rest). ' +
    'Ranges come from the leave-one-out residuals (P25 ' + v.loo_q25 + ', P75 ' + v.loo_q75 + ', P10 ' + v.loo_q10 + ', P90 ' + v.loo_q90 + ' in log points); ' +
    'their actual held-out coverage is ' + (Number(v.coverage_p25_p75) * 100).toFixed(1) + '% in P25–P75 (nominal 50%) and ' + (Number(v.coverage_p10_p90) * 100).toFixed(1) + '% in P10–P90 (nominal 80%).</p>' +
    '<p>Token odds: logistic on all ' + v.odds.n + ' protocols (' + v.odds.tokened + ' record a token, base rate ' + v.odds.base_rate + '), ' +
    '5-fold Brier <b>' + Number(v.odds.cv_brier).toFixed(3) + '</b> against a base-rate Brier of ' + Number(v.odds.base_brier).toFixed(3) + '. ' +
    'Implied value = TVL × fitted multiple × token odds; the P25–P75 and P10–P90 columns are the same value at the range ends.</p>';

  // ---- controls
  var catSel = document.getElementById('cat');
  var cats = [];
  DATA.rows.forEach(function (r) { if (r.category && cats.indexOf(r.category) === -1) cats.push(r.category); });
  cats.sort().forEach(function (c) { var o = document.createElement('option'); o.value = c; o.textContent = c; catSel.appendChild(o); });
  var qEl = document.getElementById('q'); qEl.value = state.q;
  catSel.value = state.category;
  document.getElementById('verdict').value = state.verdict;
  document.getElementById('list').value = state.list;
  document.getElementById('sort').value = state.sort;
  qEl.addEventListener('input', function () { state.q = qEl.value; render(); });
  catSel.addEventListener('change', function () { state.category = catSel.value; render(); });
  document.getElementById('verdict').addEventListener('change', function (e) { state.verdict = e.target.value; render(); });
  document.getElementById('list').addEventListener('change', function (e) { state.list = e.target.value; render(); });
  document.getElementById('sort').addEventListener('change', function (e) { state.sort = e.target.value; render(); });
  document.getElementById('tab-board').addEventListener('click', function () { setTab('board'); });
  document.getElementById('tab-live').addEventListener('click', function () { setTab('live'); });
  document.getElementById('csv').addEventListener('click', function () { download('token-radar-board.csv', 'text/csv', toCsv(filtered())); });
  document.getElementById('json').addEventListener('click', function () { download('token-radar-board.json', 'application/json', JSON.stringify(filtered(), null, 2)); });

  function setTab(tab) {
    state.tab = tab;
    document.getElementById('tab-board').classList.toggle('on', tab === 'board');
    document.getElementById('tab-live').classList.toggle('on', tab === 'live');
    document.getElementById('controls').style.display = tab === 'board' ? '' : 'none';
    document.getElementById('live').hidden = tab !== 'live';
    main.style.display = tab === 'board' ? '' : 'none';
    if (tab === 'live') renderLive();
  }
  function download(name, type, text) {
    var blob = new Blob([text], { type: type });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
  }
  function toCsv(rows) {
    var head = 'rank,slug,name,category,tvl_usd,change_7d,age_days,fitted_multiple,token_odds,implied_value_usd,p25_usd,p75_usd,verdict,usd_per_hour';
    var lines = rows.map(function (r) {
      return [r.rank, r.slug, '"' + String(r.name).replace(/"/g, '""') + '"', r.category || '', r.tvl_usd, r.change_7d === null ? '' : r.change_7d, r.age_days === null ? '' : r.age_days, r.model.fitted_multiple, r.model.token_odds, r.implied_value_usd, r.model.p25_usd, r.model.p75_usd, r.model.verdict, r.model.usd_per_hour].join(',');
    });
    return head + '\n' + lines.join('\n') + '\n';
  }

  // ---- board
  var sortKey = {
    implied: function (r) { return r.implied_value_usd; },
    tvl: function (r) { return r.tvl_usd; },
    odds: function (r) { return r.model.token_odds; },
    multiple: function (r) { return r.model.fitted_multiple; },
    perhour: function (r) { return r.model.usd_per_hour; },
    growth: function (r) { return r.change_7d === null ? -9999 : r.change_7d; }
  };
  function filtered() {
    var q = state.q.trim().toLowerCase();
    var rows = DATA.rows.filter(function (r) {
      if (q && r.name.toLowerCase().indexOf(q) === -1 && r.slug.toLowerCase().indexOf(q) === -1) return false;
      if (state.category !== 'all' && r.category !== state.category) return false;
      if (state.verdict !== 'all' && r.model.verdict !== state.verdict) return false;
      if (state.list === 'watchlist' && !isWatchlist(r)) return false;
      if (state.list === 'early' && isWatchlist(r)) return false;
      return true;
    });
    var key = sortKey[state.sort] || sortKey.implied;
    rows.sort(function (a, b) { return (key(b) - key(a)) * (state.dir === -1 ? 1 : -1) || (a.slug < b.slug ? -1 : 1); });
    return rows;
  }
  // A row is on the watchlist when the board marks it so.
  function isWatchlist(r) { return r.watchlist === true; }

  function render() {
    var rows = filtered();
    document.getElementById('count').textContent = rows.length + ' of ' + DATA.rows.length + ' rows';
    if (rows.length === 0) { main.innerHTML = '<p class="empty">No protocol matches this filter.</p>'; main.setAttribute('aria-busy', 'false'); return; }
    var html = '<table><thead><tr>' +
      '<th data-k="implied">#</th><th>Name</th><th>Category</th><th class="num" data-k="tvl">TVL</th>' +
      '<th class="num" data-k="growth">7d</th><th class="num">Age</th><th class="num" data-k="multiple">Multiple</th>' +
      '<th class="num" data-k="odds">Odds</th><th class="num" data-k="implied">Implied</th><th class="num">P25–P75</th>' +
      '<th>Verdict</th><th class="num" data-k="perhour">$/hr</th></tr></thead><tbody>';
    rows.forEach(function (r) {
      var g = r.change_7d === null ? 'n/a' : '<span class="' + (r.change_7d >= 0 ? 'pos' : 'neg') + '">' + fmtPct(r.change_7d) + '</span>';
      html += '<tr data-slug="' + esc(r.slug) + '"' + (state.selected === r.slug ? ' class="sel"' : '') + '>' +
        '<td class="num">' + r.rank + '</td>' +
        '<td class="name">' + esc(r.name) + ' <span class="slug">' + esc(r.slug) + '</span>' + (isWatchlist(r) ? ' <span class="watch">WATCHLIST</span>' : '') + '</td>' +
        '<td>' + esc(r.category || '—') + '</td>' +
        '<td class="num">' + fmtUsd(r.tvl_usd) + '</td><td class="num">' + g + '</td><td class="num">' + fmtAge(r.age_days) + '</td>' +
        '<td class="num">' + Number(r.model.fitted_multiple).toFixed(2) + '×</td>' +
        '<td class="num">' + (Number(r.model.token_odds) * 100).toFixed(0) + '%</td>' +
        '<td class="num"><b>' + fmtUsd(r.implied_value_usd) + '</b></td>' +
        '<td class="num">' + fmtUsd(r.model.p25_usd) + ' – ' + fmtUsd(r.model.p75_usd) + '</td>' +
        '<td><span class="badge ' + r.model.verdict + '">' + r.model.verdict + '</span></td>' +
        '<td class="num">' + (r.model.usd_per_hour >= 0 ? '$' : '−$') + Math.abs(Number(r.model.usd_per_hour)).toFixed(0) + '</td></tr>';
    });
    html += '</tbody></table>';
    main.innerHTML = html;
    main.setAttribute('aria-busy', 'false');
    main.querySelectorAll('th[data-k]').forEach(function (th) {
      th.addEventListener('click', function () {
        var k = th.getAttribute('data-k');
        if (state.sort === k) state.dir = -state.dir; else { state.sort = k; state.dir = -1; }
        document.getElementById('sort').value = state.sort;
        render();
      });
    });
    main.querySelectorAll('tr[data-slug]').forEach(function (tr) {
      tr.addEventListener('click', function () { state.selected = tr.getAttribute('data-slug'); render(); renderDetail(); document.getElementById('detail').scrollIntoView({ block: 'nearest' }); });
    });
  }

  // ---- detail
  function renderDetail() {
    var box = document.getElementById('detail');
    var r = state.selected ? bySlug(state.selected) : null;
    if (!r) { box.hidden = true; box.innerHTML = ''; return; }
    var m = r.model;
    var steps = r.farm_steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('');
    box.hidden = false;
    var back = document.body.classList.contains('focused-detail') ? '<p class="back"><a href="index.html">&larr; Back to the full board</a></p>' : '';
    box.innerHTML = back +
      '<h2>#' + r.rank + ' — ' + esc(r.name) + ' <span class="slug">' + esc(r.slug) + '</span></h2>' +
      '<p class="meta">' + esc(r.category || 'No category') + ' · ' + esc(r.chains.join(', ') || 'no chain listed') + ' · TVL ' + fmtFull(r.tvl_usd) +
      ' · 7d ' + fmtPct(r.change_7d) + ' · listed ' + (r.listed_at || 'unknown') + ' (' + fmtAge(r.age_days) + ') · records no token</p>' +
      '<div class="terms">' +
      '<div><b>' + fmtFull(r.implied_value_usd) + '</b><span>implied value</span></div>' +
      '<div><b>' + fmtFull(m.p25_usd) + ' – ' + fmtFull(m.p75_usd) + '</b><span>P25–P75 range</span></div>' +
      '<div><b>' + fmtFull(m.p10_usd) + ' – ' + fmtFull(m.p90_usd) + '</b><span>P10–P90 range</span></div>' +
      '<div><b>' + (Number(m.token_odds) * 100).toFixed(1) + '%</b><span>token odds (peers’ record)</span></div>' +
      '<div><b>' + Number(m.fitted_multiple).toFixed(3) + '×</b><span>fitted multiple</span></div>' +
      '<div><b>' + Number(m.size_term).toFixed(2) + ' / ' + Number(m.category_effect).toFixed(2) + '</b><span>size term / category effect (log pts)</span></div>' +
      '<div><b>' + Number(m.age_term).toFixed(2) + ' / ' + Number(m.growth_term).toFixed(2) + '</b><span>age term / growth term (log pts)</span></div>' +
      '<div><b><span class="badge ' + m.verdict + '">' + m.verdict + '</span> $' + Number(m.usd_per_hour).toFixed(0) + '/hr</b><span>farm verdict · expected ' + fmtFull(m.expected_payoff_usd) + ' · cost ' + fmtFull(m.total_cost_usd) + ' · ' + Number(m.total_hours).toFixed(1) + ' hrs</span></div>' +
      '</div>' +
      '<h3>How to farm it — six steps, each with its cost and time</h3><ol>' + steps + '</ol>' +
      '<p class="meta">Expected payoff assumes the airdrop pool is 5% of implied value and one farmer captures 0.01% of that pool. Costs are gas plus the capital’s 4.5% yearly opportunity cost over the hold. A verdict of FARM clears $25 an hour after costs.</p>';
  }

  // ---- live tokens
  function renderLive() {
    var el = document.getElementById('live');
    var html = '<table><thead><tr><th>Name</th><th>Category</th><th class="num">TVL</th><th class="num">Market cap</th><th class="num">mcap / TVL</th><th class="num">7d</th></tr></thead><tbody>';
    DATA.live.forEach(function (r) {
      var ratio = (r.mcap_usd !== null && r.tvl_usd > 0) ? (r.mcap_usd / r.tvl_usd).toFixed(2) + '×' : 'n/a';
      html += '<tr><td class="name">' + esc(r.name) + ' <span class="slug">' + esc(r.slug) + '</span></td><td>' + esc(r.category || '—') + '</td>' +
        '<td class="num">' + fmtUsd(r.tvl_usd) + '</td><td class="num">' + fmtUsd(r.mcap_usd) + '</td><td class="num">' + ratio + '</td>' +
        '<td class="num">' + fmtPct(r.change_7d) + '</td></tr>';
    });
    html += '</tbody></table><p class="meta" style="color:var(--muted)">These protocols already trade. They are the training set the multiple was fitted on — shown at the market’s own value, never ranked against the pre-token board.</p>';
    el.innerHTML = html;
  }

  if (params.get('slug')) document.body.classList.add('focused-detail');
  render();
  if (state.selected) renderDetail();
})();
