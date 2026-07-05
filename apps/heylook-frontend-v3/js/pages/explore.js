// Placeholder -- replaced by the real explore page build.
import { createPage } from '../page.js';
import { createEl } from '../utils.js';

export default createPage({
  setup(ctx) {
    ctx.el.append(createEl('div', { class: 'empty-state' }, ['explore page is being built.']));
  },
});
