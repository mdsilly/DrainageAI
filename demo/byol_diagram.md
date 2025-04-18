```mermaid
%%{init: {'theme': 'dark'}}%%
%%{init: {'theme': 'dark'}}%%
graph TB
    subgraph "Data Sources"
        A[Multispectral Imagery] 
        S[SAR Imagery]
    end
    
    subgraph "BYOL Pretraining"
        AUG1[View 1 Augmentation]
        AUG2[View 2 Augmentation]
        
        ON_ENC[Online Encoder]
        ON_PROJ[Online Projector]
        PRED[Predictor]
        
        TG_ENC[Target Encoder]
        TG_PROJ[Target Projector]
        
        EMA[EMA Update]
        
        BYOL_LOSS[BYOL Loss]
        
        A --> AUG1 & AUG2
        S --> AUG1 & AUG2
        
        AUG1 --> ON_ENC
        AUG2 --> ON_ENC
        ON_ENC --> ON_PROJ
        ON_PROJ --> PRED
        
        AUG1 --> TG_ENC
        AUG2 --> TG_ENC
        TG_ENC --> TG_PROJ
        
        PRED --> BYOL_LOSS
        TG_PROJ --> BYOL_LOSS
        
        BYOL_LOSS --> EMA
        EMA --> TG_ENC
        EMA --> TG_PROJ
    end
    
    subgraph "Fine-tuning with Few Labels"
        LABELED[Labeled Data]
        ENCODER[Pretrained Encoder]
        HEAD[Prediction Head]
        FINETUNE[Fine-tuning Loss]
        
        LABELED --> ENCODER
        ENCODER --> HEAD
        HEAD --> FINETUNE
    end
    
    subgraph "Inference Pipeline"
        F[Fine-tuned Model]
        G[Probability Map]
        H[Thresholding]
        I[Skeletonization]
        J[Vectorization]
        K[Drainage Lines]
        L[GIS Integration]
        M[ArcGIS]
        N[QGIS]
    end
    
    # Connect the components
    ON_ENC --> ENCODER
    ENCODER --> F
    F --> G --> H --> I --> J --> K --> L
    L --> M & N
    
    style A fill:#2C3E50,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style S fill:#5D3F6A,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    
    style AUG1 fill:#8E44AD,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style AUG2 fill:#8E44AD,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style ON_ENC fill:#8E44AD,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style ON_PROJ fill:#8E44AD,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style PRED fill:#8E44AD,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style TG_ENC fill:#9B59B6,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style TG_PROJ fill:#9B59B6,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style EMA fill:#9B59B6,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style BYOL_LOSS fill:#8E44AD,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    
    style LABELED fill:#2980B9,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style ENCODER fill:#2980B9,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style HEAD fill:#2980B9,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style FINETUNE fill:#2980B9,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    
    style F fill:#16A085,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style G fill:#16A085,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style H fill:#34495E,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style I fill:#34495E,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style J fill:#34495E,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style K fill:#27AE60,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style L fill:#34495E,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style M fill:#E74C3C,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
    style N fill:#E74C3C,stroke:#FFFFFF,stroke-width:2px,color:#FFFFFF
```

This Mermaid diagram shows the complete BYOL workflow for DrainageAI:

1. **Data Sources**: Multispectral and SAR imagery
2. **BYOL Pretraining**:
   - Two augmented views of the same image are created
   - Online network (encoder + projector + predictor) processes both views
   - Target network (encoder + projector) processes both views
   - BYOL loss is calculated between online predictions and target projections
   - Target network is updated via Exponential Moving Average (EMA)
3. **Fine-tuning with Few Labels**:
   - Pretrained encoder is used as feature extractor
   - New prediction head is trained on limited labeled data
4. **Inference Pipeline**:
   - Fine-tuned model generates probability maps
   - Post-processing converts predictions to vector drainage lines
   - Results are integrated with GIS software

The key advantage of this approach is that it works with extremely limited labeled data (as few as 5 labeled examples) by leveraging the power of self-supervised learning on unlabeled data.
