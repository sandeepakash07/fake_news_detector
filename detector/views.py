from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Article
from .ml_model.train_model import predict_fake_news
import requests
from bs4 import BeautifulSoup

def index(request):
    """
    Render the home page with the form to input news text or URL.
    """
    recent_articles = Article.objects.order_by('-created_at')[:5]
    return render(request, 'detector/index.html', {'recent_articles': recent_articles})

def analyze(request):
    """
    Analyze news text submitted by the user.
    """
    if request.method == 'POST':
        title = request.POST.get('title', '')
        content = request.POST.get('content', '')
        
        if not content:
            messages.error(request, 'Please provide content for analysis.')
            return redirect('detector:index')
        
        # Make prediction
        prediction, probability = predict_fake_news(content)
        
        if prediction is None:
            messages.error(request, 'Failed to make prediction. Please ensure the model is properly trained.')
            return redirect('detector:index')
        
        # Save article and prediction
        article = Article(
            title=title,
            content=content,
            is_fake=bool(prediction),
            fake_probability=probability
        )
        article.save()
        
        return render(request, 'detector/result.html', {
            'article': article,
            'probability': probability * 100,  # Convert to percentage
        })
    
    return redirect('detector:index')

def analyze_url(request):
    """
    Fetch content from a URL and analyze it.
    """
    if request.method == 'POST':
        url = request.POST.get('url', '')
        
        if not url:
            messages.error(request, 'Please provide a URL for analysis.')
            return redirect('detector:index')
        
        try:
            # Fetch content from URL
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else url
            
            # Extract article content (this is a simple implementation)
            # In a real system, you'd use a more sophisticated content extraction method
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text() for p in paragraphs])
            
            if not content:
                messages.error(request, 'Could not extract content from the URL.')
                return redirect('detector:index')
            
            # Make prediction
            prediction, probability = predict_fake_news(content)
            
            if prediction is None:
                messages.error(request, 'Failed to make prediction. Please ensure the model is properly trained.')
                return redirect('detector:index')
            
            # Save article and prediction
            article = Article(
                title=title,
                content=content,
                url=url,
                is_fake=bool(prediction),
                fake_probability=probability
            )
            article.save()
            
            return render(request, 'detector/result.html', {
                'article': article,
                'probability': probability * 100,  # Convert to percentage
            })
            
        except requests.exceptions.RequestException as e:
            messages.error(request, f'Error fetching URL: {e}')
            return redirect('detector:index')
    
    return redirect('detector:index')