document.getElementById('prediction-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const sqft = document.getElementById('sqft').value;
    const bedrooms = document.getElementById('bedrooms').value;
    const bathrooms = document.getElementById('bathrooms').value;
    const age = document.getElementById('age').value;
    const location = document.getElementById('location').value;

    const placeholderText = document.getElementById('placeholder-text');
    const resultContent = document.getElementById('result-content');
    const priceDisplay = document.getElementById('predicted-price');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ sqft, bedrooms, bathrooms, age, location })
        });

        const data = await response.json();

        if (response.ok) {
            placeholderText.classList.add('hidden');
            resultContent.classList.remove('hidden');

            priceDisplay.innerText = `$${data.prediction.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        console.error('API Error:', error);
        alert('Could not establish connection to the backend server.');
    }
});