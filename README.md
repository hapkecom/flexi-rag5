flexi-rag3
==========

Start DB

    # start
    ./rag-build-image.sh
    ./all-start.sh    #  (includes sudo)
        # to stop log output: CTRL-C
    
    # see containers
    docker ps    
        
    # test: go to Perplexica start: http://localhost:3000/
    
    # stop at the end
    ./all-stop.sh


Services:
- weaviate: vector database
    - ports 8080 + 50051
- rag: RAG app that crawls, indexes and searchs in your content
    - uses weaviate
    - exposes Searxng-compatible on port 8000
- perplexica: frontend chat app
    - uses rag
    - port 3000



Test Searxng-compatible (rag-app) endpoint

    curl -X GET "http://localhost:8000/search?q=What+happened+at+Interleaf%3F&max_results=5" -H "accept: application/json"
    curl -X GET "http://localhost:8000/search?q=What+is+Java%3F&max_results=5" -H "accept: application/json"

    # test for hapke.com
    curl -X GET "http://localhost:8000/search?q=What+happened+with+Hapke%3F&max_results=5" -H "accept: application/json"
    
    # test for https://unec.edu.az/application/uploads/2014/12/pdf-sample.pdf
    curl -X GET "http://localhost:8000/search?q=What+is+a+PDF%3F&max_results=5" -H "accept: application/json"


