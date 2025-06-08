# Architectural Element Classification System

A sophisticated AI-powered application for classifying architectural elements in floor plans, designed for construction estimating and building analysis. The system combines three advanced classification approaches: Spatial Fusion (rule-based), Deep Learning CNN, and Large Language Model analysis.

## Features

### Core Capabilities
- **Multi-Method Classification**: Three distinct AI approaches for maximum accuracy
- **PDF Vector Extraction**: Direct processing of architectural PDF files
- **Google Drive Integration**: Seamless access to cloud-stored files
- **Manual Training System**: Build custom training datasets interactively
- **Construction Export**: Generate JSON data optimized for estimating workflows
- **Performance Analytics**: Comprehensive metrics and comparison tools

### Classification Types
- 🧱 **Walls**: Structural barriers and load-bearing elements
- 🚪 **Doors**: Openings with swing indicators and access points
- 🪟 **Windows**: Exterior openings and glazing elements
- 🏠 **Rooms**: Spaces defined by text labels and boundaries
- 🔧 **Fixtures**: Plumbing, HVAC, and built-in equipment
- 📏 **Dimensions**: Measurement annotations and dimension lines
- 🏷️ **Labels**: Text annotations and identification markers

## Installation

### Requirements
- Python 3.8 or higher
- OpenAI API key (for LLM classification)
- Google Cloud credentials (optional, for Google Drive integration)

### Dependencies
```bash
pip install streamlit>=1.28.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install pillow>=10.0.0
pip install scikit-learn>=1.3.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install opencv-python>=4.8.0
pip install PyMuPDF>=1.23.0
pip install openai>=1.0.0
pip install google-api-python-client>=2.100.0
pip install google-auth-httplib2>=0.2.0
pip install google-auth-oauthlib>=1.2.0
pip install seaborn>=0.12.0
```

### Quick Start
1. Clone or download all files to your project directory
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```
4. Run the application: `streamlit run app_standalone.py`

## File Structure

```
architectural-classifier/
├── app_standalone.py                 # Main Streamlit application
├── data_loader_standalone.py         # PDF and JSON processing
├── google_drive_integration.py       # Cloud storage integration
├── performance_metrics_standalone.py # Analytics and reporting
├── classifiers/
│   ├── spatial_fusion_classifier_standalone.py    # Rule-based classifier
│   ├── deep_learning_standalone.py                # CNN classifier
│   └── llm_based_standalone.py                   # OpenAI GPT classifier
├── README.md                         # This documentation
└── requirements.txt                  # Python dependencies (create manually)
```

## Usage Guide

### Basic Workflow
1. **Upload Files**: Load PDF floor plan and JSON vector data
2. **Select Methods**: Choose classification approaches to use
3. **Run Analysis**: Process elements with selected classifiers
4. **Review Results**: Examine classifications and performance metrics
5. **Export Data**: Download results for construction estimating

### Advanced Features

#### Manual Training
- Upload PDF files independently for training
- Select and classify elements interactively
- Adjust element boundaries for precision
- Build persistent training datasets
- Export training data for model improvement

#### Google Drive Integration
- Connect to Google Drive with OAuth2 authentication
- Browse and search PDF files directly
- Load existing training data from cloud storage
- Merge remote and local training datasets

#### Model Deep Dive
- Detailed explanations of each classification method
- Architecture diagrams and processing pipelines
- Performance comparisons and recommendations
- Technical specifications and limitations

## Classification Methods

### 1. Spatial Fusion Classifier
**Type**: Rule-based geometric analysis
**Strengths**: Fast, interpretable, no training required
**Best For**: Standard architectural layouts, real-time processing

**Key Features**:
- Geometric pattern recognition
- Spatial relationship analysis
- Architectural convention matching
- Context-aware classification

### 2. Deep Learning CNN
**Type**: Convolutional Neural Network
**Strengths**: Visual pattern recognition, complex geometry handling
**Best For**: Non-standard layouts, detailed visual analysis

**Key Features**:
- Vector-to-image conversion
- Multi-layer feature extraction
- Context-enhanced processing
- Fallback rule validation

### 3. LLM-Based Classification
**Type**: Large Language Model (GPT-4o)
**Strengths**: Human-like reasoning, complex pattern understanding
**Best For**: Highest accuracy requirements, complex layouts

**Key Features**:
- Contextual element analysis
- Architectural domain expertise
- Batch processing optimization
- Detailed reasoning capture

## API Configuration

### OpenAI Setup
1. Create account at [OpenAI](https://platform.openai.com/)
2. Generate API key in your dashboard
3. Set environment variable: `OPENAI_API_KEY="your_key"`
4. Monitor usage and billing in OpenAI console

### Google Drive Setup (Optional)
1. Create project in [Google Cloud Console](https://console.cloud.google.com/)
2. Enable Google Drive API
3. Create OAuth 2.0 credentials (Desktop application)
4. Configure credentials in the application interface
5. Complete OAuth flow for drive access

## Performance Optimization

### Speed Optimization
- Use Spatial Fusion for fastest processing (<1 second)
- Enable Deep Learning for balanced speed/accuracy
- Reserve LLM for highest accuracy needs (3-5 seconds)

### Accuracy Optimization
- Combine multiple methods for consensus classification
- Use manual training to improve rule-based classification
- Process high-resolution PDFs for better vector extraction
- Provide context through surrounding elements

### Resource Management
- Monitor OpenAI API usage and costs
- Use batch processing for large datasets
- Cache LLM responses for repeated elements
- Optimize image resolution for CNN processing

## Data Formats

### Input Requirements
- **PDF Files**: Vector-based architectural drawings (not scanned images)
- **JSON Files**: Element metadata with coordinates and properties
- **Supported Elements**: Lines, polylines, rectangles, circles, text, arcs

### Output Formats
- **Classification Results**: Element-by-element classifications
- **Construction JSON**: Structured data for estimating software
- **Performance Metrics**: Accuracy, speed, and confidence scores
- **Training Data**: Labeled datasets for model improvement

## Troubleshooting

### Common Issues
1. **PDF Processing Errors**: Ensure PDF contains vector data, not scanned images
2. **API Rate Limits**: Monitor OpenAI usage and implement delays if needed
3. **Memory Issues**: Process large datasets in smaller batches
4. **Google Drive Auth**: Verify OAuth credentials and scopes

### Performance Issues
1. **Slow Processing**: Reduce batch sizes or disable unused classifiers
2. **Low Accuracy**: Increase training data or adjust confidence thresholds
3. **High API Costs**: Use caching and optimize batch processing

### File Format Issues
1. **JSON Validation**: Ensure proper element structure and required fields
2. **Coordinate Systems**: Verify coordinate consistency between PDF and JSON
3. **Text Encoding**: Use UTF-8 encoding for all text files

## Advanced Configuration

### Classifier Settings
```python
# Spatial Fusion Classifier
wall_min_length = 50          # Minimum wall length threshold
door_size_range = (20, 100)   # Door size constraints
window_aspect_range = (1.5, 4.0)  # Window aspect ratio limits

# Deep Learning CNN
image_size = (224, 224)       # Input image dimensions
batch_size = 32               # Processing batch size
confidence_threshold = 0.6    # Minimum confidence for acceptance

# LLM Classifier
api_batch_size = 25           # Elements per API call
max_retries = 3               # Retry attempts for failed calls
temperature = 0.1             # Response randomness (lower = more consistent)
```

### Performance Tuning
```python
# Speed vs Accuracy Trade-offs
speed_optimized = {
    'use_spatial_fusion': True,
    'use_deep_learning': False,
    'use_llm': False
}

accuracy_optimized = {
    'use_spatial_fusion': True,
    'use_deep_learning': True,
    'use_llm': True,
    'consensus_threshold': 0.6
}
```

## Contributing

### Adding New Classifiers
1. Implement classifier interface in `classifiers/` directory
2. Add classification method to main application
3. Update performance metrics integration
4. Include documentation and examples

### Improving Existing Methods
1. Enhance rule sets in Spatial Fusion classifier
2. Retrain CNN with additional architectural data
3. Optimize LLM prompts for better accuracy
4. Add new element types and classifications

## License and Usage

This software is designed for architectural analysis and construction estimating. Please ensure compliance with applicable licensing terms for all dependencies, including OpenAI API usage terms and Google Drive API policies.

## Support

For technical issues, feature requests, or implementation guidance, refer to the inline documentation and method docstrings throughout the codebase. The application includes comprehensive error handling and user feedback for most common scenarios.