function testAPI() {
    const wordInput = document.getElementById('apiWord');
    const word = wordInput.value.trim();
    if (!word) {
        showToast('Please enter a word', 'warning');
        return;
    }

    const btn = document.querySelector('button[onclick="testAPI()"]');
    const originalContent = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Testing...';
    btn.disabled = true;

    fetch('/api/correct', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ word: word })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById('apiResponse').textContent = JSON.stringify(data, null, 2);
            document.getElementById('apiResult').style.display = 'block';
            gsap.from('#apiResult', { opacity: 0, y: 20, duration: 0.5 });
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('API request failed', 'error');
        })
        .finally(() => {
            btn.innerHTML = originalContent;
            btn.disabled = false;
        });
}

function addWord() {
    const wordInput = document.getElementById('newWord');
    const word = wordInput.value.trim();
    if (!word) {
        showToast('Please enter a word', 'warning');
        return;
    }

    const btn = document.querySelector('button[onclick="addWord()"]');
    btn.disabled = true;

    fetch('/api/add-word', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ word: word })
    })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('addResult');
            resultDiv.style.display = 'block';
            resultDiv.className = data.success ? 'alert alert-success border-0 rounded-4' : 'alert alert-warning border-0 rounded-4';
            resultDiv.textContent = data.message;
            gsap.from(resultDiv, { opacity: 0, scale: 0.9, duration: 0.4 });
            if (data.success) wordInput.value = '';
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Request failed', 'error');
        })
        .finally(() => {
            btn.disabled = false;
        });
}

function showToast(message, type = 'info') {
    // Simple toast fallback if no library is used
    console.log(`[${type.toUpperCase()}] ${message}`);
}
