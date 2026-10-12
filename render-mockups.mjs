import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
const browser=await chromium.launch({headless:true});
const page=await browser.newPage({viewport:{width:1600,height:900},deviceScaleFactor:1});
const source=`file:///${resolve('ui-mockups.html').replaceAll('\\','/')}`;
await mkdir('样式参考图',{recursive:true});
for(const [theme,name] of [['command','01-深色AI中控台'],['hud','02-轻量HUD迎宾界面'],['enterprise','03-企业级智能迎宾终端']]){await page.goto(`${source}?theme=${theme}`);await page.screenshot({path:`样式参考图/${name}.png`});}
await browser.close();
