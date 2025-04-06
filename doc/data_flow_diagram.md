graph TD
    subgraph Initialization
        A[Start] --> B(Create MagicCardDetector);
        B --> C{Load Reference Data?};
        C -- Yes --> D[Read Reference Images / Hashes];
        C -- No --> E;
        D --> E[Load Test Images];
    end

    subgraph Image Processing Loop [For each Test Image]
        E --> F(Test Image);
        F --> G[Contour Image];
        G -- Contours --> H[Segment Image];
        H -- Potential Card Contours --> I{Characterize Contour};
        I -- Valid Contour --> J[Get Bounding Quad];
        J -- Bounding Quad --> K[Perspective Transform];
        K -- Straightened Card Image --> L[Create CardCandidate Object];
        L --> M(List of Card Candidates);
    end

    subgraph Recognition Loop [For each Card Candidate]
        M --> N(Card Candidate Image);
        N --> O[Calculate phash];
        O -- phash --> P{Compare phash with Reference Hashes};
        P -- Best Match --> Q[Update CardCandidate: Name, Score];
    end

    subgraph Post-Processing & Output
        M --> R[Mark Fragments in Candidate List];
        R -- Filtered Candidate List --> S{Generate Output?};
        S -- Plot --> T[Plot Recognized Cards on Image];
        T --> U[Save Output Image];
        S -- Print List --> V[Print Recognized Card Names];
        U --> W[End];
        V --> W;
        S -- No Output --> W;
    end

    %% Data Stores
    DS1[Reference Image Hashes]:::datastore;
    DS2[Test Image Data]:::datastore;
    DS3[Card Candidate Data]:::datastore;

    %% Link Data Stores
    D --> DS1;
    E --> DS2;
    L --> DS3;
    Q --> DS3;
    R --> DS3;
    P --> DS1;
    G --> DS2;
    K --> DS2;


    classDef datastore fill:#f9f,stroke:#333,stroke-width:2px;