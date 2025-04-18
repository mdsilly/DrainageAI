graph TD
    subgraph "FixMatch Semi-Supervised Learning"
        A[Training Data] --> B[Labeled Data<br>~5-10 examples]
        A --> C[Unlabeled Data<br>~100s of images]
        
        B --> D[Supervised Learning]
        C --> E[Weak Augmentation]
        C --> F[Strong Augmentation]
        
        E --> G[Generate Pseudo-Labels]
        F --> H[Apply Pseudo-Labels]
        
        G --> H
        
        D --> I[Combined Loss Function]
        H --> I
        
        I --> J[Trained Model]
        
        J --> K[Inference on New Data]
    end
    
    style A fill:#f9d5e5,stroke:#333,stroke-width:2px
    style B fill:#d5e8d4,stroke:#333,stroke-width:2px
    style C fill:#dae8fc,stroke:#333,stroke-width:2px
    style D fill:#d5e8d4,stroke:#333,stroke-width:2px
    style E fill:#dae8fc,stroke:#333,stroke-width:2px
    style F fill:#dae8fc,stroke:#333,stroke-width:2px
    style G fill:#ffe6cc,stroke:#333,stroke-width:2px
    style H fill:#ffe6cc,stroke:#333,stroke-width:2px
    style I fill:#f8cecc,stroke:#333,stroke-width:2px
    style J fill:#e1d5e7,stroke:#333,stroke-width:2px
    style K fill:#e1d5e7,stroke:#333,stroke-width:2px
```

This Mermaid diagram illustrates the FixMatch semi-supervised learning approach used in DrainageAI:

1. **Data Splitting**: We use a small amount of labeled data and a large amount of unlabeled data
2. **Supervised Learning**: Traditional supervised learning on the labeled data
3. **Pseudo-Labeling**: 
   - Apply weak augmentation to unlabeled data and generate pseudo-labels
   - Apply strong augmentation to unlabeled data and train using the pseudo-labels
4. **Combined Training**: Merge the supervised loss and the pseudo-label loss
5. **Result**: A model that leverages both labeled and unlabeled data

The key innovation is using confidence thresholding to ensure only high-quality pseudo-labels are used for training, which significantly improves performance with limited labeled data.

For the presentation, this diagram should be converted to a PNG or SVG image using a Mermaid renderer or a similar tool.
