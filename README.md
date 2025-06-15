# Personalized Shopper: E-commerce Recommendation System

A scalable e-commerce recommendation system built with Apache Spark, Delta Lake, and Neural Collaborative Filtering (NCF) for processing 10M+ user interactions.

## Features

- Scalable data processing with Apache Spark
- ACID transactions and time travel with Delta Lake
- Neural Collaborative Filtering for personalized recommendations
- Real-time and batch processing capabilities
- Model training and serving pipeline
- Monitoring and evaluation metrics

## Project Structure

```
├── data/                  # Data storage directory
├── notebooks/            # Jupyter notebooks for analysis
├── src/
│   ├── data/            # Data processing modules
│   ├── models/          # NCF model implementation
│   ├── training/        # Model training pipeline
│   └── serving/         # Model serving and inference
├── tests/               # Unit tests
├── config/              # Configuration files
└── requirements.txt     # Python dependencies
```

## Prerequisites

- Python 3.8+
- Apache Spark 3.2+
- Delta Lake 2.0+
- PyTorch 1.9+
- Jupyter Notebook

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Jupyter Notebook:
```bash
jupyter notebook
```

## Usage

1. Data Processing:
```bash
python src/data/process_data.py
```

2. Model Training:
```bash
python src/training/train_model.py
```

3. Generate Recommendations:
```bash
python src/serving/generate_recommendations.py
```

## License

MIT License