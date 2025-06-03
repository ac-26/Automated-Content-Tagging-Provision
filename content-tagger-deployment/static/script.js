// Auto-detect if running locally or on Hugging Face
const API_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://localhost:7860' 
    : '';

// DOM Elements
const textInput = document.getElementById('textInput');
const maxTagsInput = document.getElementById('maxTags');
const minScoreInput = document.getElementById('minScore');
const includeScoresCheckbox = document.getElementById('includeScores');
const generateBtn = document.getElementById('generateBtn');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const tagsContainer = document.getElementById('tagsContainer');
const processingTimeSpan = document.getElementById('processingTime');
const tagCountSpan = document.getElementById('tagCount');

// Event Listeners
generateBtn.addEventListener('click', generateTags);
textInput.addEventListener('input', validateInput);

// Enter key to generate tags
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        generateTags();
    }
});

// Validate input on load
validateInput();

// Functions
function validateInput() {
    const text = textInput.value.trim();
    generateBtn.disabled = text.length < 10;
}

async function generateTags() {
    const text = textInput.value.trim();
    
    if (text.length < 10) {
        showError('Please enter at least 10 characters of text.');
        return;
    }
    
    // Show loading, hide others
    showSection('loading');
    
    try {
        // Prepare request data
        const requestData = {
            text: text,
            max_tags: parseInt(maxTagsInput.value),
            min_score: parseFloat(minScoreInput.value),
            include_scores: includeScoresCheckbox.checked
        };
        
        // Choose endpoint based on whether we want detailed scores
        const endpoint = includeScoresCheckbox.checked ? '/tag/detailed' : '/tag';
        
        // Make API request
        const response = await fetch(`${API_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError(`Failed to generate tags: ${error.message}`);
    }
}

function displayResults(data) {
    // Clear previous results
    tagsContainer.innerHTML = '';
    
    // Display tags
    if (data.tags && Array.isArray(data.tags)) {
        if (typeof data.tags[0] === 'object') {
            // Detailed format with scores
            data.tags.forEach(tagObj => {
                createTagElement(tagObj.tag, tagObj.score);
            });
        } else {
            // Simple format
            data.tags.forEach((tag, index) => {
                const score = data.scores ? data.scores[index] : null;
                createTagElement(tag, score);
            });
        }
    }
    
    // Update stats
    processingTimeSpan.textContent = data.processing_time.toFixed(3);
    tagCountSpan.textContent = data.tag_count;
    
    // Show results section
    showSection('results');
}

function createTagElement(tagText, score = null) {
    const tagDiv = document.createElement('div');
    tagDiv.className = 'tag';
    
    // Add score-based styling
    if (score !== null) {
        if (score >= 0.8) {
            tagDiv.classList.add('high-score');
        } else if (score >= 0.7) {
            tagDiv.classList.add('medium-score');
        }
    }
    
    // Add tag text
    const tagTextSpan = document.createElement('span');
    tagTextSpan.textContent = tagText;
    tagDiv.appendChild(tagTextSpan);
    
    // Add score if available
    if (score !== null && includeScoresCheckbox.checked) {
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'tag-score';
        scoreSpan.textContent = score.toFixed(2);
        tagDiv.appendChild(scoreSpan);
    }
    
    tagsContainer.appendChild(tagDiv);
}

function showSection(section) {
    // Hide all sections
    loadingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    
    // Show requested section
    switch(section) {
        case 'loading':
            loadingSection.classList.remove('hidden');
            break;
        case 'results':
            resultsSection.classList.remove('hidden');
            break;
        case 'error':
            errorSection.classList.remove('hidden');
            break;
    }
}

function showError(message) {
    const errorMessage = errorSection.querySelector('.error-message');
    errorMessage.textContent = message;
    showSection('error');
}

// Add some example texts for quick testing
const exampleTexts = {
    tech: "Machine learning algorithms are transforming how we process big data. Python libraries like TensorFlow and PyTorch make it easier to build neural networks for deep learning applications.",
    marketing: "Creating social media posts is a great way to hone your content writing skills. Since posts are typically very short, snappy, and quick, you can easily try out different styles of writing.",
    medical: "The patient presented with acute respiratory symptoms including persistent cough and shortness of breath. Blood tests revealed elevated white blood cell count."
};

// Optional: Add example buttons (you can add this to HTML if wanted)
console.log('Tagger Frontend Ready!');
console.log('Tip: Press Ctrl+Enter in the text area to generate tags quickly.');