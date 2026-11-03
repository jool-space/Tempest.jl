// Draws the README banner: two rows of cat's-eye vortices with the name between them, one file per GitHub theme.
//
//   cd docs/banner && npm install && node build.mjs
//
// The flow is Stuart's solution for a shear layer rolled up into a row of vortices, doubled into two layers
// of opposite sign: the middle band moves right, the outer bands move left, and the eyes turn between them.
// Its streamlines are drawn as contours of the stream function, each eye's loops shaded toward a brighter green
// at its centre, and short streaks move along every line at the flow's own speed there.
//
// GitHub shows a README image through <img>, so the files hold no script and load no font: the motion is CSS
// keyframes, the colours are fixed per theme, and the name is drawn as outlines.

import { contours } from 'd3-contour';
import { Blob, Face, Font, Buffer, Variation, shape } from 'harfbuzzjs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = join(HERE, '..', 'src', 'assets');

const W = 1280, H = 360, MID = H / 2, LANE = 40;
const CELL = 4;                   // drawing unit in px; paths are written in these units, scaled up by the frame
const SPACING = 9, V0 = 22;       // streak spacing in drawing units; streak speed in px/s in the free bands
const THEMES = {
  dark:  { line: '#52B33A', hot: '#A1F266', ink: '#E8F4D2' },
  light: { line: '#5BAF3D', hot: '#2E8F1A', ink: '#389826' },
};
const FONT = {
  url: 'https://raw.githubusercontent.com/google/fonts/95f4904fc8bcf26d3420fe315560c96417c6dec7/ofl/anybody/Anybody%5Bwdth,wght%5D.ttf',
  file: join(HERE, '.cache', 'Anybody.ttf'),
  variations: { wght: 800, wdth: 135 },
};

const g2 = v => (Math.round(v * 100) / 100).toString();

// Douglas-Peucker: drop vertices that sit within tol of the line through their neighbours.
function simplify(pts, tol) {
  const keep = new Uint8Array(pts.length);
  keep[0] = keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [i0, i1] = stack.pop(), [ax, ay] = pts[i0], [bx, by] = pts[i1], dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
    let best = -1, bi = -1;
    for (let i = i0 + 1; i < i1; i++) {
      const [px, py] = pts[i];
      const d = L2 ? Math.abs(dy * (px - ax) - dx * (py - ay)) / Math.sqrt(L2) : Math.hypot(px - ax, py - ay);
      if (d > best) { best = d; bi = i; }
    }
    if (best > tol) { keep[bi] = 1; stack.push([i0, bi], [bi, i1]); }
  }
  return pts.filter((_, i) => keep[i]);
}

// The streamlines of two rows of cat's eyes, 90 px above and below the middle.
// lam: eye spacing in px. Cc: near 1 the eyes are long and flat, larger they are round.
// rings: loops per eye, evenly spaced in distance from the centre; the outermost sits a fraction edge of the way
// in from the eye's boundary, in psi. Outside the eyes, evenly spaced lines with the nearest a fraction gap of a
// step outside each boundary.
function stuart({ lam, Cc, rings, edge = 0.5 / rings, gap = 0.5 }) {
  const k = 2 * Math.PI / lam, s = Math.sqrt(Cc * Cc - 1), a = 90 * k, cell = CELL, my = 2, mx = Math.round(lam / cell);
  // Contour grid points per drawing unit. A 4 px grid made the smallest loops visibly polygonal.
  const fine = 4;
  const nx = W / cell + 2 * mx, ny = H / cell + 2 * my, NX = nx * fine, NY = ny * fine;
  const psiS = (X, e) => Math.log(Cc * Math.cosh(e) - s * Math.cos(X));
  const psi = (X, Y) => { const kx = k * X, e = -(Y - MID) * k; return psiS(kx + Math.PI, e + a) - psiS(kx, e - a) - e; };
  const vals = new Float64Array(NX * NY);
  for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) vals[j * NX + i] = psi(((i + 0.5) / fine - mx) * cell, ((j + 0.5) / fine - my) * cell);
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const cont = contours().size([NX, NY]);
  // Screen velocity is (-dpsi/dY, dpsi/dX); a line's streaks move forward along it when that velocity points along it.
  const h = 0.05, vel = (X, Y) => [-(psi(X, Y + h) - psi(X, Y - h)), psi(X + h, Y) - psi(X - h, Y)];
  const eye = [], open = [];
  // keep: 'closed' or 'open' rings only; row: closed rings above (-1) or below (+1) the middle only.
  function trace(t, keep, row = 0, depth = 0) {
    for (const pg of cont.contour(vals, t).coordinates) for (const r0 of pg) {
      const r = simplify(r0.map(p => [p[0] / fine, p[1] / fine]), 0.015).map(p => [+g2(p[0]), +g2(p[1])]);
      let len = 0, x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity, sy = 0;
      for (let i = 0; i < r.length; i++) {
        const [x, y] = r[i];
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); sy += y;
        if (i) len += Math.hypot(x - r[i - 1][0], y - r[i - 1][1]);
      }
      if (x1 < mx - 1 || x0 > nx - mx + 1 || len < 1) continue;   // wholly outside the frame
      const closed = x0 > 1 && x1 < nx - 1 && y0 > 1 && y1 < ny - 1, Y = (sy / r.length - my) * cell;
      if ((keep === 'closed' && !closed) || (keep === 'open' && closed)) continue;
      if (closed && row && Math.sign(Y - MID) !== row) continue;
      // Sum the flow along the ring's segments inside the frame. Open rings also run along the grid border, which says nothing about the flow.
      let along = 0, spd = 0, vis = 0;
      for (let i = 1; i < r.length; i++) {
        const qx = (r[i][0] + r[i - 1][0]) / 2, qy = (r[i][1] + r[i - 1][1]) / 2;
        if (qx < mx || qx > nx - mx || qy < my || qy > ny - my) continue;
        const v = vel((qx - mx) * cell, (qy - my) * cell), dl = Math.hypot(r[i][0] - r[i - 1][0], r[i][1] - r[i - 1][1]);
        along += v[0] * (r[i][0] - r[i - 1][0]) + v[1] * (r[i][1] - r[i - 1][1]);
        spd += Math.hypot(v[0], v[1]) / (2 * h * k) * dl;
        vis += dl;
      }
      if (!vis) continue;   // runs only along the grid border, outside the frame
      // speed: the line's mean flow speed, 1 = the free bands far from the eyes.
      const ring = { d: 'M' + r.map(p => p[0] + ' ' + p[1]).join('L'), len, fwd: along > 0, speed: spd / vis, depth };
      (closed ? eye : open).push(ring);
    }
  }
  // Each row's eye centre and saddle lie on vertical lines through X = 0 and X = lam/2, where psi is stationary along Y.
  const statAt = (X, Yc) => {
    let best = null; const ys = [];
    for (let y = Yc - 40; y <= Yc + 40; y += 0.25) ys.push([y, psi(X, y)]);
    for (let i = 1; i < ys.length - 1; i++) if ((ys[i][1] - ys[i - 1][1]) * (ys[i + 1][1] - ys[i][1]) <= 0) best = ys[i];
    return best;
  };
  const stat = (X, Yc) => statAt(X, Yc)[1];
  const rowsAt = [[-1, MID - 90, 0, lam / 2], [1, MID + 90, lam / 2, 0]];   // row, Y, X of eye centre, X of saddle
  let sTop = 0;
  for (const [row, Yc, Xc, Xs] of rowsAt) {
    const [yc, c] = statAt(Xc, Yc), sv = stat(Xs, Yc);
    if (row < 0) sTop = sv;
    // Loops evenly spaced in distance from the centre, not in psi: psi grows with the square of the distance,
    // so even psi steps crowd the loops outward and leave the centre empty.
    const Lr = r => (psi(Xc, yc - r) + psi(Xc, yc + r)) / 2, target = sv + edge * (c - sv);
    let r0 = 0, r1 = 80;
    for (let it = 0; it < 50; it++) { const rm = (r0 + r1) / 2; if ((Lr(rm) - target) * (c - sv) > 0) r0 = rm; else r1 = rm; }
    for (let j = 0; j < rings; j++) trace(Lr(r0 * (rings - j) / rings), 'closed', row, (j + 0.5) / rings);
  }
  // The spacing outside is fitted so the nearest line sits gap of a step off both boundaries:
  // the levels between them must read the same from either end.
  const cTop = stat(0, MID - 90), dir = Math.sign(sTop - cTop);   // which side of the top boundary lies outside the eyes
  const S = Math.abs(sTop), Dt = Math.abs((cTop - sTop) / rings);
  const Db = 2 * S / (Math.max(1, Math.round(2 * S / Dt - 2 * gap)) + 2 * gap);
  const n0 = Math.floor(Math.min((lo - sTop) * dir, (hi - sTop) * dir) / Db) - 1, n1 = Math.ceil(Math.max((lo - sTop) * dir, (hi - sTop) * dir) / Db) + 1;
  for (let n = n0; n <= n1; n++) { const t = sTop + dir * (gap + n) * Db; if (t > lo && t < hi) trace(t, 'open'); }
  return { eye, open, tr: `translate(${-mx * cell} ${-my * cell}) scale(${cell})` };
}

// Colours mixed in Oklab, as CSS color-mix(in oklab, a, b p%) would.
function toLab(hex) {
  const lin = c => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  const [r, g, b] = [1, 3, 5].map(i => lin(parseInt(hex.slice(i, i + 2), 16) / 255));
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s, 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s];
}
function fromLab([L, a, b]) {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3, m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3, s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  const rgb = [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s, -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s, -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
  const enc = x => Math.round(255 * Math.max(0, Math.min(1, x > 0.0031308 ? 1.055 * x ** (1 / 2.4) - 0.055 : 12.92 * x)));
  return '#' + rgb.map(x => enc(x).toString(16).toUpperCase().padStart(2, '0')).join('');
}
const mix = (a, b, p) => { const A = toLab(a), B = toLab(b); return fromLab(A.map((x, i) => x + (B[i] - x) * p / 100)); };

// The name, shaped and outlined, sized and placed as on the sketch page: at most 74 px, at most 640 px wide,
// centred on the middle, with the em box centred vertically (CSS dominant-baseline: central).
async function wordmark(text = 'Tempest', fit = 640, maxSize = 74) {
  if (!existsSync(FONT.file)) {
    const res = await fetch(FONT.url);
    if (!res.ok) throw new Error(`downloading ${FONT.url}: ${res.status}`);
    await mkdir(dirname(FONT.file), { recursive: true });
    await writeFile(FONT.file, new Uint8Array(await res.arrayBuffer()));
  }
  const face = new Face(new Blob(await readFile(FONT.file)));
  const font = new Font(face);
  font.setVariations(Object.entries(FONT.variations).map(([tag, value]) => new Variation(tag, value)));
  const buf = new Buffer();
  buf.addText(text);
  buf.guessSegmentProperties();
  shape(font, buf);
  const glyphs = buf.getGlyphInfos(), pos = buf.getGlyphPositions();
  const adv = pos.reduce((t, p) => t + p.xAdvance, 0);
  const size = Math.min(maxSize, fit * face.upem / adv), k = size / face.upem;
  const { ascender, descender } = font.hExtents();
  const base = MID + (ascender + descender) / 2 * k;
  let x = W / 2 - adv * k / 2, d = '';
  const num = v => (Math.round(v * 100) / 100).toString();
  glyphs.forEach((g, i) => {
    const ox = x + pos[i].xOffset * k, oy = base - pos[i].yOffset * k;
    // glyphToPath gives commands in font units, y up: map every coordinate pair into the frame.
    d += font.glyphToPath(g.codepoint).replace(/([MLQCZ])([^MLQCZ]*)/g, (_, cmd, args) => {
      const n = args.trim() ? args.trim().split(/[\s,]+/).map(Number) : [];
      return cmd + n.map((v, j) => num(j % 2 ? oy - v * k : ox + v * k)).join(' ');
    });
    x += pos[i].xAdvance * k;
  });
  return d;
}

// Every path is stored once in <defs> and drawn through <use>: the faint line, then three streak layers
// (a tail, a stronger half tail, a dot at the head), all moving together at the line's flow speed.
function banner(ST, C, name) {
  const all = ST.eye.concat(ST.open), keyframes = new Map();
  const ref = new Map(all.map((r, i) => [r, `p${i}`]));
  const tone = r => r.depth ? mix(C.line, C.hot, Math.round(r.depth * 85)) : C.line;
  const anim = to => { const v = to.toFixed(3); if (!keyframes.has(v)) keyframes.set(v, `k${keyframes.size}`); return keyframes.get(v); };
  const lines = ST.open.map(r => `<use href="#${ref.get(r)}" stroke-opacity=".4"/>`).join('') +
    ST.eye.map(r => `<use href="#${ref.get(r)}" stroke="${tone(r)}" stroke-opacity="${(0.4 + 0.45 * r.depth).toFixed(2)}"/>`).join('');
  const streaks = all.map(r => {
    const p = r.len / Math.max(1, Math.round(r.len / SPACING)), Ls = Math.min(3, p * 0.5), sg = r.fwd ? -1 : 1;
    const dur = (p * CELL / (Math.max(r.speed, 0.1) * V0)).toFixed(2);
    const seg = (len, off, op, w) => `<use href="#${ref.get(r)}" stroke="${tone(r)}" stroke-dasharray="${len ? len.toFixed(3) : 0} ${(p - len).toFixed(4)}" ` +
      `stroke-dashoffset="${off.toFixed(3)}" stroke-opacity="${op}" stroke-width="${w}" style="animation:${anim(off + sg * p)} ${dur}s linear infinite"/>`;
    return seg(Ls, 0, .22, .5) + seg(Ls / 2, r.fwd ? -Ls / 2 : 0, .45, .55) + seg(0, r.fwd ? -Ls : 0, 1, .6);
  }).join('');
  const css = [...keyframes].map(([to, k]) => `@keyframes ${k}{to{stroke-dashoffset:${to}}}`).join('') +
    '@media (prefers-reduced-motion:reduce){*{animation:none!important}}';
  // Fades: out toward the left and right edges, and out toward the name's lane and the top and bottom edges.
  const hfade = `<linearGradient id="hG" gradientUnits="userSpaceOnUse" x1="90" y1="0" x2="1190" y2="0"><stop offset="0" stop-color="#000"/><stop offset=".2" stop-color="#fff"/><stop offset=".8" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>` +
    `<mask id="h" maskUnits="userSpaceOnUse" x="0" y="0" width="${W}" height="${H}"><rect width="${W}" height="${H}" fill="url(#hG)"/></mask>`;
  const vfade = `<linearGradient id="vG" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="${H}"><stop offset="0" stop-color="#000"/><stop offset=".12" stop-color="#fff"/>` +
    `<stop offset="${(MID - LANE - 22) / H}" stop-color="#fff"/><stop offset="${(MID - LANE + 6) / H}" stop-color="#000"/><stop offset="${(MID + LANE - 6) / H}" stop-color="#000"/><stop offset="${(MID + LANE + 22) / H}" stop-color="#fff"/>` +
    `<stop offset=".88" stop-color="#fff"/><stop offset="1" stop-color="#000"/></linearGradient>` +
    `<mask id="v" maskUnits="userSpaceOnUse" x="0" y="0" width="${W}" height="${H}"><rect width="${W}" height="${H}" fill="url(#vG)"/></mask>`;
  return `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Tempest">` +
    `<title>Tempest</title><style>${css}</style><defs>${hfade}${vfade}${all.map(r => `<path id="${ref.get(r)}" d="${r.d}"/>`).join('')}</defs>` +
    `<g mask="url(#h)"><g mask="url(#v)">` +
    `<g transform="${ST.tr}" fill="none" stroke="${C.line}"><g stroke-width=".3">${lines}</g><g stroke-linecap="round">${streaks}</g></g></g></g>` +
    `<path fill="${C.ink}" d="${name}"/></svg>\n`;
}

const ST = stuart({ lam: W / 6, Cc: 1.12, rings: 5 });
const name = await wordmark();
await mkdir(OUT, { recursive: true });
for (const [theme, C] of Object.entries(THEMES)) {
  const file = join(OUT, `tempest-banner-${theme}.svg`), svg = banner(ST, C, name);
  await writeFile(file, svg);
  console.log(`${file}: ${(svg.length / 1024).toFixed(0)} KB, ${ST.eye.length} loops, ${ST.open.length} open lines`);
}
