// Placeholder -- replaced by the real notebook page build.
import { createPage } from '../page.js';
import { createEl } from '../utils.js';

export default createPage({
  setup(ctx) {
    ctx.el.append(createEl('div', { class: 'empty-state' }, ['notebook page is being built.']));
  },
});
