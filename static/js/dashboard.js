document.addEventListener('DOMContentLoaded', function () {
    // Regional Chart
    const regionalCtx = document.getElementById('regionalChart');
    if (regionalCtx) {
        new Chart(regionalCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                datasets: [{
                    label: 'North India',
                    data: [65000, 72000, 85000, 92000, 105000, 120000],
                    borderColor: '#FF9933',
                    backgroundColor: 'rgba(255, 153, 51, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'South India',
                    data: [55000, 68000, 79000, 88000, 99000, 115000],
                    borderColor: '#138808',
                    backgroundColor: 'rgba(19, 136, 8, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'East India',
                    data: [35000, 42000, 51000, 62000, 73000, 85000],
                    borderColor: '#000080',
                    backgroundColor: 'rgba(0, 0, 128, 0.1)',
                    tension: 0.4,
                    fill: true
                }, {
                    label: 'West India',
                    data: [48000, 56000, 67000, 78000, 89000, 102000],
                    borderColor: '#FF3366',
                    backgroundColor: 'rgba(255, 51, 102, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: 'rgba(255,255,255,0.7)', font: { family: 'Satoshi' } } }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'rgba(255,255,255,0.5)' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: 'rgba(255,255,255,0.5)' }
                    }
                }
            }
        });
    }

    // Language Chart
    const languageCtx = document.getElementById('languageChart');
    if (languageCtx) {
        new Chart(languageCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Hindi', 'English', 'Tamil', 'Telugu', 'Bengali', 'Others'],
                datasets: [{
                    data: [35, 25, 12, 10, 8, 10],
                    backgroundColor: [
                        '#FF9933',
                        '#138808',
                        '#000080',
                        '#FF3366',
                        '#00FF88',
                        '#9933FF'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: 'rgba(255,255,255,0.7)', padding: 20 }
                    }
                },
                cutout: '70%'
            }
        });
    }

    // Initialize Counters
    if (typeof animateCounter === 'function') {
        document.querySelectorAll('.counter').forEach(counter => {
            const target = parseInt(counter.getAttribute('data-target'));
            animateCounter(counter.id, target);
        });
    }
});
