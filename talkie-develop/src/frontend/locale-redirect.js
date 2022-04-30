/*
This file is part of Talkie -- text-to-speech browser extension button.
<https://joelpurra.com/projects/talkie/>

Copyright (c) 2016, 2017, 2018, 2019, 2020, 2021 Joel Purra <https://joelpurra.com/>

Talkie is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Talkie is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Talkie.  If not, see <https://www.gnu.org/licenses/>.
*/

const pageName = document.location.pathname.split("/").pop().split(".").shift();

/* eslint-disable dot-notation */
const globalExtension = ("chrome" in this ? this["chrome"] : ("browser" in this ? this["browser"] : null));
/* eslint-enable dot-notation */

// NOTE: might be able to use @@ui_locale, but would miss out on locale directoy fallback detection logic in the browser.
const locale = globalExtension.i18n.getMessage("extensionLocale");
const urlWithLocale = document.location.href.replace(`/src/${pageName}/${pageName}.html`, `/dist/${pageName}.${locale}.html`);

document.location.replace(urlWithLocale);
