document.addEventListener('DOMContentLoaded', () => {
  const count = document.querySelector('#selection-count');
  const updateCount = () => { if (count) count.textContent = document.querySelectorAll('.slot input:checked').length; };
  document.querySelectorAll('.slot input').forEach(input => input.addEventListener('change', updateCount));
  updateCount();
  document.querySelectorAll('[data-confirm]').forEach(form => form.addEventListener('submit', event => { if (!window.confirm(form.dataset.confirm)) event.preventDefault(); }));
  const menu = document.querySelector('.mobile-menu');
  if (menu) menu.addEventListener('click', () => document.body.classList.toggle('menu-open'));
});
