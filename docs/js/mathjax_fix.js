// https://github.com/mathjax/MathJax/issues/2312#issuecomment-2440036455
// https://docs.mathjax.org/en/stable/input/tex/extensions/mathtools.html
window.MathJax = {
    startup: {
    ready() {
      MathJax.startup.defaultReady();
      MathJax.startup.document.inputJax[0].preFilters.add(({math}) => {
        if (math.math.match(/\\\\/)) {
          math.math = `\\displaylines{${math.math}}`;
        }
      });
    }
  },
  loader: {load: ['[tex]/mathtools']},
  tex: {packages: {'[+]': ['mathtools']}}
};