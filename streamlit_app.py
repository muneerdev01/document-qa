from openai import OpenAI
import traceback

def ask_ai(query, context, history):
    try:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            }
        ]

        for h in history[-5:]:
            messages.append(h)

        messages.append({
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{query}
"""
        })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        st.error(f"OpenAI Error: {str(e)}")
        st.code(traceback.format_exc())
        return None
