import streamlit as st
import json
import pandas as pd
import time
from rank import score_candidate, score_semantic_embedding, generate_reasoning, JD_TEXT

st.set_page_config(page_title="Redrob AI Recruiter", page_icon="👔", layout="wide")

# 8. Better Title
st.title("Redrob AI Recruiter")
st.subheader("Intelligent Candidate Ranking Engine")

st.markdown("---")

# 1. Allow users to paste or upload their own Job Description
st.markdown("### Upload Job Description")

jd_file = st.file_uploader(
    "Upload Job Description (.txt, .docx)",
    type=["txt", "docx"],
    label_visibility="collapsed"
)

default_jd = JD_TEXT
if jd_file:
    try:
        if jd_file.name.endswith(".docx"):
            import docx2txt
            default_jd = docx2txt.process(jd_file)
        else:
            default_jd = jd_file.read().decode("utf-8")
        st.toast("Successfully loaded Job Description from file.")
    except Exception as e:
        st.error(f"Error reading file: {e}")

st.markdown("##### Or paste/edit below:")
jd_input = st.text_area(
    "Paste Job Description",
    value=default_jd,
    height=220,
    label_visibility="collapsed"
)

# 2. Upload Candidate File
st.markdown("### Upload Candidate Dataset")
uploaded_file = st.file_uploader(
    "Upload Candidate JSON",
    type=["json", "jsonl"],
    label_visibility="collapsed"
)

st.markdown("---")

@st.cache_data
def load_sample_candidates():
    with open("sample_candidates.json", "r", encoding="utf-8") as f:
        return json.load(f)

if st.button("Run Ranking", type="primary", use_container_width=True):
    start_time = time.time()
    
    # Load Candidates
    candidates = []
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".jsonl"):
                for line in uploaded_file:
                    line_text = line.decode('utf-8').strip()
                    if line_text:
                        candidates.append(json.loads(line_text))
            else:
                candidates = json.load(uploaded_file)
            st.toast(f"Successfully loaded {len(candidates)} candidates from upload.")
        except Exception as e:
            st.error(f"Failed to parse uploaded file: {e}")
    else:
        try:
            candidates = load_sample_candidates()
            st.info("No file uploaded. Using sample_candidates.json as fallback.")
        except Exception as e:
            st.error(f"Failed to load sample candidates: {e}")
            
    if candidates:
        # 6. Progress Bar
        progress_text = "Scoring candidates..."
        progress_bar = st.progress(0, text=progress_text)
        
        scored, disq = [], 0
        total_cands = len(candidates)
        
        for i, cand in enumerate(candidates):
            sc = score_candidate(cand)
            if sc['disqualified']:
                disq += 1
            else:
                scored.append((cand, sc))
            
            if i % max(1, total_cands // 20) == 0:
                progress_bar.progress(min(int((i / total_cands) * 80), 80), text=f"Scoring candidates ({i}/{total_cands})...")
                
        # Initial Sort
        scored.sort(key=lambda x: (-x[1]['final_score'], x[0].get('candidate_id', '')))
        
        # Semantic Re-ranking
        progress_bar.progress(85, text="Running semantic re-ranking on top candidates...")
        top_subset = scored[:1000]
        score_semantic_embedding(top_subset, jd_input)
        
        # Normalize scores to have a maximum of 0.999
        max_score = max((sc['final_score'] for cand, sc in scored), default=1.0)
        if max_score > 0:
            scale = 0.999 / max_score
            for cand, sc in scored:
                sc['final_score'] *= scale

        # Re-sort all scored candidates since top_subset scores were modified
        scored.sort(key=lambda x: (-x[1]['final_score'], x[0].get('candidate_id', '')))
        
        progress_bar.progress(100, text="Finalizing results...")
        
        # 3. Show WHY a candidate ranked highly
        results = []
        detailed_data = []
        
        # Process ALL candidates for the CSV download
        for rank, (cand, sc) in enumerate(scored, start=1):
            reason = generate_reasoning(cand, sc, rank)
            cand_name = cand.get("profile", {}).get("anonymized_name", cand.get("candidate_id", "Unknown"))
            
            results.append({
                "Rank": rank,
                "Candidate": cand_name,
                "Score": round(sc["final_score"], 4),
                "Reason": reason
            })
            
            # Only keep detailed data for the top 100 to prevent the UI from freezing
            if rank <= 100:
                detailed_data.append((cand, sc, reason))
            
        df = pd.DataFrame(results)
        end_time = time.time()
        runtime = f"{end_time - start_time:.2f}s"
        
        # Clear progress bar
        time.sleep(0.5)
        progress_bar.empty()
        
        # Save to session_state
        st.session_state['has_results'] = True
        st.session_state['df'] = df
        st.session_state['detailed_data'] = detailed_data
        st.session_state['scored_count'] = len(scored)
        st.session_state['top_score'] = top_subset[0][1]['final_score'] if top_subset else 0
        st.session_state['runtime'] = runtime


# Only display results if they exist in session_state
if st.session_state.get('has_results'):
    # 5. Nice Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Qualified Candidates", st.session_state['scored_count'])
    col2.metric("Top Score", f"{st.session_state['top_score']:.4f}")
    col3.metric("Runtime", st.session_state['runtime'])
    
    st.markdown("---")
    
    # Show DataFrame
    st.dataframe(st.session_state['df'], use_container_width=True)
    
    # 4. Download Results
    csv = st.session_state['df'].to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="ranked_candidates.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # 7. Candidate Details
    st.markdown("### Candidate Details")
    for i, (cand, sc, reason) in enumerate(st.session_state['detailed_data']):
        cand_name = cand.get("profile", {}).get("anonymized_name", cand.get("candidate_id", "Unknown"))
        with st.expander(f"#{i+1} - {cand_name} (Score: {sc['final_score']:.4f})"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Matched Concepts:**")
                st.write(", ".join(sc.get('matched_concepts', [])) if sc.get('matched_concepts') else "None")
                st.markdown("**Power Combo:**")
                st.write(sc.get('power_combos', ["None"])[0] if sc.get('power_combos') else "None")
                
                st.markdown("**GitHub / OSS:**")
                signals = cand.get("redrob_signals", {})
                gh = signals.get('github_activity_score', -1)
                st.write(f"Active (Score: {gh:.0f}/100)" if gh > 65 else "No significant activity / Unknown")
                
                st.markdown("**Availability:**")
                st.write("Score: " + str(round(sc.get('availability_score', 0), 2)))
                
            with c2:
                st.markdown("**Career Notes:**")
                if sc.get('career_notes'):
                    for n in sc.get('career_notes'):
                        st.write(f"- {n}")
                else:
                    st.write("No special career notes.")
                    
                st.markdown("**Education:**")
                edu = cand.get('education', [])
                if edu:
                    for e in edu[:2]:
                        st.write(f"- {e.get('degree')} in {e.get('major')} from {e.get('school', 'Unknown')} ({e.get('end_date', 'Present')})")
                else:
                    st.write("No education details available.")
                    
            st.markdown("**Reasoning:**")
            st.info(reason)
