# ManipulaPy Documentation Redesign

## Summary

Redesign the ManipulaPy Read the Docs site as an Editorial Robotics Lab: a precise, image-led documentation experience for robotics developers, researchers, and students. Keep Sphinx, `pydata-sphinx-theme`, every published route, search, theme switching, and the existing information architecture. Recompose the homepage and reskin the documentation system without altering the public API or technical contracts.

## Design Direction

The approved direction is Editorial Robotics Lab.

- `DESIGN_VARIANCE: 7`: asymmetric homepage composition with disciplined inner documentation pages.
- `MOTION_INTENSITY: 4`: brief hierarchy-driven entrances and one-time reveals only.
- `VISUAL_DENSITY: 5`: compact enough for technical reference, with more breathing room around learning content.
- Light-first presentation with a complete dark-mode equivalent. A page uses one coherent theme at a time.
- Cold neutral palette with silver surfaces, graphite text, and one restrained teal accent.
- Self-hosted Space Grotesk headings, system-sans prose, and self-hosted JetBrains Mono code.
- A consistent 6px corner radius. Pill shapes are reserved for controls whose interaction requires them.

## Homepage Structure

The homepage becomes a concise entry point rather than a long reference page.

1. **Editorial hero**
   - Asymmetric text and media composition.
   - Headline: "Move from equations to motion."
   - Supporting copy stays below 20 words.
   - One primary action labeled "Start building" linking to Getting Started.
   - A compact copyable installation command is adjacent supporting content, not a competing CTA.
   - Use one generated robotics-specific hero image with useful alt text.

2. **Learning paths**
   - Four intent-based routes: learn robotics, load a robot, plan motion, and use accelerated backends.
   - Each route uses a distinct composition within one asymmetric grid, not four identical cards.
   - Links preserve existing destinations and labels where those labels appear in primary navigation.

3. **Motion gallery**
   - Reuse selected committed robot trajectory and workspace GIFs as real product media.
   - Provide descriptive alt text and stable dimensions to prevent layout shift.
   - No labels overlaid on images and no decorative photo-credit captions.

4. **Backend overview**
   - Explain NumPy, CuPy, PyTorch, and JAX without invented benchmarks or fake precision.
   - Use a compact capability composition rather than a long row-divided table.
   - Link to the existing Compute Backends guide and backend API page.

5. **API entry points**
   - End with direct routes into kinematics, dynamics, planning, control, perception, and reference material.
   - Use grouped columns with sparse dividers, not repeated cards.

Remove the badge wall, the inline duplicate page contents block, long worked examples, release notes, license prose, and contribution prose from the homepage. Preserve that information through links to its existing canonical pages or repository files.

## Documentation Shell

- Keep the PyData left navigation, content column, right table of contents, search, theme switcher, source link, and responsive navigation behavior.
- Condense the top navigation so it remains one line at 1024px and above.
- Use the project name as the primary brand label instead of the full versioned documentation title.
- Improve heading hierarchy, prose measure, link treatment, code blocks, copy buttons, tables, admonitions, API signatures, and notebook output spacing.
- On screens below 768px, collapse every asymmetric homepage section into a strict single column with stable media dimensions and comfortable side padding.
- Preserve all route slugs, anchor IDs generated from unchanged inner-page headings, primary navigation labels, and Read the Docs configuration.

## Assets and Motion

- Generate one hero asset at the composition's final aspect ratio and store the optimized web output under the existing Sphinx static assets directory.
- Reuse existing committed GIFs for supporting motion. Do not generate fake product interfaces or decorative SVG illustrations.
- Keep JavaScript optional. The complete document must render visibly and remain navigable when scripts fail.
- Restrict animation to transforms and opacity. Use the existing vendored runtime only if it materially reduces code; otherwise use CSS plus `IntersectionObserver`.
- Animate the hero once on initial load and reveal major media groups once as they approach the viewport.
- Disable all nonessential movement under `prefers-reduced-motion: reduce`.

## Accessibility and Failure Behavior

- Meet WCAG AA contrast for prose, controls, code, focus rings, and search UI in both themes.
- Preserve keyboard operation for search, navigation, theme switching, source links, and copy buttons.
- Use semantic headings and links. Custom homepage markup must not create empty links or duplicate headings.
- Reserve image dimensions and provide alt text. If media cannot load, surrounding content and navigation still communicate the page's purpose.
- Preserve PyData's native empty and no-result search behavior rather than replacing it.

## Implementation Boundaries

- Main changes belong in `docs/source/index.rst`, `docs/source/_static/custom.css`, and the existing small motion script/configuration surface.
- Theme configuration may change only where needed for branding, navigation composition, fonts, or asset registration.
- Do not change Python package behavior, documentation routes, API names, form fields, analytics identifiers, legal text, or Read the Docs deployment targets.
- Do not introduce React, Tailwind, a second Sphinx theme, or a new client-side framework.

## Verification

- Build Sphinx HTML without new warnings and verify PDF and ePub builders still complete.
- Capture desktop and mobile screenshots of the homepage and representative guide/API pages in light and dark modes.
- Test search, code-copy controls, theme switching, keyboard focus, mobile navigation, source links, and reduced-motion behavior.
- Check internal links, image paths, image alt text, and stable image dimensions.
- Run mechanical scans for visible em-dash and en-dash characters, duplicate CTA labels with equivalent intent, excessive eyebrow labels, and repeated section layouts.
- Run the applicable `design-taste-frontend` pre-flight checklist and correct every failure before completion.
- Confirm no unrelated worktree files are modified or committed.
