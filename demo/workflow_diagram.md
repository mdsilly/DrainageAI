graph LR
    A[Multispectral Imagery] --> B[Calculate Spectral Indices]
    B --> C[NDVI]
    B --> D[NDMI]
    B --> E[MSAVI2]
    C --> F[CNN Model]
    D --> F
    E --> F
    A --> F
    F --> G[Probability Map]
    G --> H[Thresholding]
    H --> I[Skeletonization]
    I --> J[Vectorization]
    J --> K[Drainage Lines]
    K --> L[GIS Integration]
    L --> M[ArcGIS]
    L --> N[QGIS]
    
    style A fill:#f9d5e5,stroke:#333,stroke-width:2px
    style B fill:#eeeeee,stroke:#333,stroke-width:2px
    style C fill:#d5e8d4,stroke:#333,stroke-width:2px
    style D fill:#d5e8d4,stroke:#333,stroke-width:2px
    style E fill:#d5e8d4,stroke:#333,stroke-width:2px
    style F fill:#dae8fc,stroke:#333,stroke-width:2px
    style G fill:#dae8fc,stroke:#333,stroke-width:2px
    style H fill:#eeeeee,stroke:#333,stroke-width:2px
    style I fill:#eeeeee,stroke:#333,stroke-width:2px
    style J fill:#eeeeee,stroke:#333,stroke-width:2px
    style K fill:#d5e8d4,stroke:#333,stroke-width:2px
    style L fill:#eeeeee,stroke:#333,stroke-width:2px
    style M fill:#f8cecc,stroke:#333,stroke-width:2px
    style N fill:#f8cecc,stroke:#333,stroke-width:2px
```

This Mermaid diagram shows the complete workflow of the DrainageAI system:

1. **Input**: Multispectral satellite imagery
2. **Preprocessing**: Calculation of spectral indices (NDVI, NDMI, MSAVI2)
3. **Model**: CNN-based detection using both imagery and indices
4. **Post-processing**: Thresholding, skeletonization, and vectorization
5. **Output**: Vector drainage lines for GIS integration

For the presentation, this diagram should be converted to a PNG or SVG image using a Mermaid renderer or a similar tool.
