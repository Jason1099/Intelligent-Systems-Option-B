document.addEventListener('DOMContentLoaded', () => {
  const links = document.querySelectorAll('.nav-link');
  const sections = document.querySelectorAll('.content');

  function show(page) {
    // switch sections
    sections.forEach(s => s.classList.toggle('active', s.id === page));
    // switch nav active
    links.forEach(a => a.classList.toggle('active', a.dataset.page === page));
  }

  // click handling
  links.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const page = link.dataset.page;
      if (document.getElementById(page)) {
        show(page);
        // update URL hash so refresh keeps the same tab
        history.replaceState(null, '', `#${page}`);
      } else {
        console.warn(`No section with id="${page}"`);
      }
    });
  });

  // open the page from URL hash on load
  const initial = location.hash.replace('#', '') || 'main';
  show(initial);
});
