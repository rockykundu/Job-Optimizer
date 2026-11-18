SCENARIOS = [
    {
        "id": "backend_engineering",
        "expected_title_pattern": r"backend|back-end|software engineer|python|django|api|platform engineer",
        "resume": """
        Backend software engineer with five years of experience building Python,
        Django and FastAPI services. Strong PostgreSQL, Redis, REST API, Docker,
        AWS, Linux, Git, automated testing and CI/CD experience. Designed reliable
        distributed services and production monitoring.
        """,
    },
    {
        "id": "frontend_engineering",
        "expected_title_pattern": r"frontend|front-end|react|javascript|typescript|web developer|ui developer",
        "resume": """
        Frontend engineer specialising in React, TypeScript, JavaScript, HTML,
        CSS, Next.js, accessible design systems, responsive web applications,
        Git, REST APIs, Jest and frontend performance optimisation.
        """,
    },
    {
        "id": "mobile_engineering",
        "expected_title_pattern": r"mobile|android|ios|flutter|react native",
        "resume": """
        Mobile application developer experienced with Flutter, Dart, Android,
        iOS, Firebase, REST APIs, Git, unit testing and publishing production
        applications to mobile app stores.
        """,
    },
    {
        "id": "data_and_ai",
        "expected_title_pattern": r"data|machine learning|artificial intelligence|\bai\b|analytics|scientist",
        "resume": """
        Data scientist and machine learning engineer using Python, pandas,
        scikit-learn, PyTorch, SQL, statistics, data analysis, deep learning,
        natural language processing, Tableau, experiment design and model
        deployment.
        """,
    },
    {
        "id": "devops_and_cloud",
        "expected_title_pattern": r"devops|site reliability|\bsre\b|cloud engineer|platform engineer|infrastructure",
        "resume": """
        DevOps and cloud engineer with Kubernetes, Docker, Terraform, AWS,
        Linux, networking, GitHub Actions, CI/CD, infrastructure as code,
        observability, incident response and production reliability experience.
        """,
    },
]
