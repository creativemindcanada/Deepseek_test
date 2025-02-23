const puppeteer = require('puppeteer');

(async () => {
  // Launch a new browser instance in headless mode
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--disable-gpu', '--no-sandbox'],
  });

  // Create a new page
  const page = await browser.newPage();

  // Navigate to the website
  await page.goto('https://www.example.com');

  // Scrape the website content
  const content = await page.content();

  // Save the results to a file
  const fs = require('fs');
  fs.writeFileSync('results.json', JSON.stringify({ content }));

  // Close the browser
  await browser.close();
})();
