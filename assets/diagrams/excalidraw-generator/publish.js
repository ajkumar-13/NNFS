// Promote a rendered hand-drawn scene into the directory the posts embed.
//
// The generators write to posts/NN-slug/diagrams/excalidraw/, which holds both
// the editable .excalidraw scene and its rendered .svg. The figure a post
// embeds lives one level up, in diagrams/. This is the step between the two,
// so `npm run build:01 && npm run publish 01` is the whole loop after an edit.
//
// This is opt-in on purpose. The clean vector SVGs one level up are the
// canonical figures; running this REPLACES one of them with the hand-drawn
// alternate under the same filename. Only .svg moves; the scenes and the
// README stay where they are.

const fs = require('fs');
const path = require('path');

const POSTS = path.join(__dirname, '..', '..', '..', 'posts');
const only = process.argv[2];               // optional: a post number, e.g. 01
// Optional: the specific figures to promote, without the .svg. Most posts want
// this — a hand-drawn mirror of a figure that already exists is an alternate,
// and only the figures the post has no clean vector for should be promoted.
const names = process.argv.slice(3).map((n) => n.replace(/\.svg$/, ''));

let copied = 0, skipped = 0;
for (const slug of fs.readdirSync(POSTS).sort()) {
  if (only && !slug.startsWith(only)) continue;
  const dia = path.join(POSTS, slug, 'diagrams');
  const work = path.join(dia, 'excalidraw');
  if (!fs.existsSync(work)) continue;

  for (const name of fs.readdirSync(work).sort()) {
    if (!name.endsWith('.svg')) continue;
    if (names.length && !names.includes(name.replace(/\.svg$/, ''))) continue;
    const from = path.join(work, name);
    const to = path.join(dia, name);
    const next = fs.readFileSync(from);
    // Skip a write that would change nothing, so mtimes stay meaningful and it
    // is obvious from the output which figures a rebuild actually moved.
    if (fs.existsSync(to) && fs.readFileSync(to).equals(next)) { skipped++; continue; }
    fs.writeFileSync(to, next);
    console.log(`  ${slug}/diagrams/${name}`);
    copied++;
  }
}

console.log(`${copied} figure(s) published, ${skipped} already current`);
if (!copied && !skipped) {
  console.error('nothing found: run a build first, e.g. npm run build:01');
  process.exit(1);
}
