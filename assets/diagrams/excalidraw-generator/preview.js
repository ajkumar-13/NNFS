// Flatten --ce-* CSS variables to literal hex so resvg can render a preview.
// The shipped SVG keeps the variables; this is for eyeballing only.
//
//   node preview.js path/to/diagram.svg out-light.png
//   node preview.js path/to/diagram.svg out-dark.png dark
const { Resvg } = require('@resvg/resvg-js');
const fs = require('fs');

const LIGHT = {
  bg: '#FAFAF7', surface: '#FFFFFF', ink: '#1A1A1A', 'ink-muted': '#5C5C5C', 'ink-subtle': '#9A9A9A',
  border: '#D9D9D4', primary: '#5B7FBF', accent: '#D98E5F', success: '#5C9E78', warn: '#B8895A',
  alert: '#C66B5E', 'neutral-1': '#EAEAE4', 'neutral-2': '#CFCFC8', 'neutral-3': '#8E8E88',
  'on-fill': '#FFFDF9', 'on-accent': '#1A1A1A', grid: 'rgba(26,26,26,0.06)',
};
const DARK = {
  bg: '#0E0F12', surface: '#16181C', ink: '#F2F2EE', 'ink-muted': '#B4B4AE', 'ink-subtle': '#6E6E68',
  border: '#2A2D33', primary: '#8BA8E0', accent: '#E8B088', success: '#7FBF9B', warn: '#D4B58A',
  alert: '#D88880', 'neutral-1': '#1F2229', 'neutral-2': '#2C3038', 'neutral-3': '#6E6E68',
  'on-fill': '#14161A', 'on-accent': '#1A1A1A', grid: 'rgba(241,238,232,0.08)',
};

function flatten(svg, pal) {
  let out = svg.replace(/var\(--ce-([a-z0-9-]+)\)/g, (m, k) => pal[k] || '#FF00FF');
  // Keep the <style> block: repo SVGs drive everything through classes. Only
  // the CSS variables and the @media dark block need flattening.
  out = out.replace(/@media \(prefers-color-scheme:\s*dark\)\s*\{[\s\S]*?\}\s*\}/g, '');
  out = out.replace(/<style><!\[CDATA\[/g, '<style>').replace(/\]\]><\/style>/g, '</style>');
  if (!/\.bg\s*\{/.test(out)) out = out.replace(/class="bg"/, `fill="${pal.bg}"`);
  return out;
}

function png(svgPath, outPath, { dark = false, width = 1100 } = {}) {
  const pal = dark ? DARK : LIGHT;
  const flat = flatten(fs.readFileSync(svgPath, 'utf8'), pal);
  const r = new Resvg(flat, {
    fitTo: { mode: 'width', value: width },
    background: pal.bg,
    font: { loadSystemFonts: true },
  });
  fs.writeFileSync(outPath, r.render().asPng());
}

module.exports = { png, flatten, LIGHT, DARK };
if (require.main === module) {
  const [, , src, out, mode] = process.argv;
  png(src, out, { dark: mode === 'dark' });
  console.log('wrote', out);
}
