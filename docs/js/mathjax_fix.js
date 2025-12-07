// reference https://github.com/mathjax/MathJax/issues/2312#issuecomment-2440036455
MathJax = {
  startup: {
    ready() {
      MathJax.startup.defaultReady();
      MathJax.startup.document.inputJax[0].preFilters.add(({math}) => {
        if (math.math.match(/\\\\/)) {
          math.math = `\\displaylines{${math.math}}`;
        }
      });
    }
  }
}

// window.MathJax = {
//   chtml: {
//     scale: 1.0,  // 统一缩放所有公式
//     matchFontHeight: false,
//     displayAlign: 'center'
//   }
// };