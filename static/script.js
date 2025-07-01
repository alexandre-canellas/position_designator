document.getElementById('predictForm').addEventListener('submit', async function(e) {
      e.preventDefault();
      const form = e.target;
      const data = Object.fromEntries(new FormData(form).entries());

      // Convert numeric fields to numbers
      for (let key in data) {
        if (!isNaN(data[key])) data[key] = Number(data[key]);
      }

      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      const json = await res.json();
      document.getElementById('result').innerHTML =
        `<p><strong>Posição:</strong> ${json.pred}</p>`;
    });