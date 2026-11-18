from __future__ import annotations
from pathlib import Path
import streamlit as st
from pypdf import PdfReader
from src.job_optimizer_engine import JobOptimizerEngine
PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Job Optimizer",
    page_icon="",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_engine() -> JobOptimizerEngine:
    return JobOptimizerEngine(PROJECT_ROOT)

# pdf checkes here 
def extract_resume_text(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise ValueError("The uploaded PDF is password-protected.") from exc
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        return "\n".join(pages).strip()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("The PDF could not be read. Please upload a valid text-based PDF.") from exc
    
# alignment sets
def alignment_label(score: float) -> str:
    if score >= 80:
        return "Strong alignment"
    if score >= 70:
        return "Good alignment"
    if score >= 60:
        return "Partial alignment"
    return "Limited alignment"

#
def render_skill_list(title: str, values: list[str], empty_message: str) -> None:
    st.markdown(f"**{title}**")
    if values:
        st.write(", ".join(values))
    else:
        st.caption(empty_message)


try:
    engine = load_engine()
except Exception as exc:
    st.error("The validated matching engine could not be loaded.")
    st.exception(exc)
    st.stop()


st.title("Job Optimizer")
st.markdown(
    "Upload a text-based PDF resume to receive ranked job matches from the "
    "validated semantic-graph model.")

with st.sidebar:
    st.header("System")
    st.success("Semantic Graph Matcher v1.1")
    # st.metric("Rankable jobs", f"{len(engine.main_job_ids):,}")
    st.caption("Local SBERT embeddings + weighted heterogeneous GNN")
    st.divider()
    st.caption(
        "Compatibility scores support job exploration. They are not hiring "
        "probabilities, employment guarantees, or recruiter decisions.")

with st.form("resume_analysis_form", clear_on_submit=False):
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf"],
        help="Use a text-based PDF. Scanned image-only PDFs are not currently supported.",
    )
    top_k = st.slider(
        "Number of job matches",
        min_value=5,
        max_value=25,
        value=10,
        step=5,
    )
    submitted = st.form_submit_button(
        "Analyze resume",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if uploaded_file is None:
        st.warning("Upload a PDF resume before starting the analysis.")
    else:
        try:
            with st.spinner("Reading and matching your resume..."):
                resume_text = extract_resume_text(uploaded_file)
                if len(resume_text.strip()) < 50:
                    raise ValueError(
                        "Not enough text could be extracted. Upload a text-based resume PDF.")
                result = engine.rank_resume(resume_text, top_k=top_k)
            st.session_state["job_optimizer_result"] = result
            st.session_state["job_optimizer_filename"] = uploaded_file.name
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("job_optimizer_result", None)
        except Exception as exc:
            st.error("Resume analysis failed. The trained model and project files should be checked.")
            st.exception(exc)
            st.session_state.pop("job_optimizer_result", None)

result = st.session_state.get("job_optimizer_result")
if result:
    st.success(
        f"Analysis completed for {st.session_state.get('job_optimizer_filename', 'resume')}."
    )
    candidate = result["candidate"]
    summary_columns = st.columns(3)
    summary_columns[0].metric("Detected skills", candidate["detected_skill_count"])
    summary_columns[1].metric("Jobs evaluated", f"{len(engine.main_job_ids):,}")
    summary_columns[2].metric("Matches displayed", len(result["matches"]))

    st.subheader("Detected canonical skills")
    if candidate["detected_skills"]:
        st.write(", ".join(candidate["detected_skills"]))
    else:
        st.warning(
            "No canonical skills were detected. Results rely primarily on semantic resume content."
        )

    st.divider()
    st.header("Ranked job matches")
    st.caption(
        "Jobs are ranked using the uncapped model output. Percentages are bounded "
        "to 0–100 only for display."
    )

    for match in result["matches"]:
        score = float(match["display_composite_score"])
        heading = (
            f"{match['rank']}. {match['title']} — {match['company']} | {score:.1f}%"
        )
        with st.expander(heading, expanded=match["rank"] == 1):
            information, analysis = st.columns([2, 1])
            with information:
                st.subheader("Job information")
                st.write(f"**Company:** {match['company']}")
                st.write(f"**Location:** {match['location']}")
                st.write(f"**Role:** {match['title']}")
                if match.get("url"):
                    st.link_button("Open original job posting", match["url"])
                with st.expander("View job description"):
                    st.write(match["description"] or "No description is available.")

                st.subheader("Skill evidence")
                if not match["structured_skill_evidence_available"]:
                    st.info(
                        "Structured skill evidence is unavailable for this job. "
                        "The compatibility score is based on semantic and graph features; "
                        "no conclusion about skill gaps is shown."
                    )
                else:
                    render_skill_list(
                        "Matched required skills",
                        match["matched_required_skills"],
                        "No required-skill overlap was detected.",
                    )
                    render_skill_list(
                        "Matched preferred skills",
                        match["matched_preferred_skills"],
                        "No preferred-skill overlap was detected.",
                    )
                    render_skill_list(
                        "Other matched skills",
                        match["matched_mentioned_skills"],
                        "No additional mentioned-skill overlap was detected.",
                    )
                    render_skill_list(
                        "Missing required skills",
                        match["missing_required_skills"],
                        "No missing required skills were identified from the available evidence.",
                    )
                    render_skill_list(
                        "Missing preferred skills",
                        match["missing_preferred_skills"],
                        "No missing preferred skills were identified from the available evidence.",
                    )

            with analysis:
                st.subheader("Match analysis")
                st.metric("Overall compatibility", f"{score:.1f}%")
                st.metric(
                    "Skills compatibility",
                    f"{float(match['display_skills_score']):.1f}%",
                )
                st.info(alignment_label(score))
                st.caption(
                    "The overall score combines semantic resume/job representations "
                    "with weighted candidate–skill–job graph information."
                )

    st.divider()
    st.warning(result["disclaimer"])
