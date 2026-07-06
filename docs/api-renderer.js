const referencePath = "api-reference.json";
const moduleNav = document.querySelector("[data-api-nav]");
const onPageNav = document.querySelector("[data-api-on-page-nav]");
const modulesRoot = document.querySelector("[data-api-root]");
const pageTitle = document.querySelector("[data-api-title]");
const pageLede = document.querySelector("[data-api-lede]");
const params = new URLSearchParams(window.location.search);
const currentModule = params.get("module") || "";

const slug = (value) =>
  String(value)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "");

const moduleShortName = (moduleName) => moduleName.split(".").pop();
const modulePage = (moduleName, hash = "") =>
  `api.html?module=${encodeURIComponent(moduleName)}${hash}`;

const text = (tagName, value, className) => {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  element.textContent = value || "";
  return element;
};

const code = (value) => {
  const element = document.createElement("code");
  element.textContent = value || "";
  return element;
};

const paragraph = (value) => text("p", value);

const codeBlock = (value) => {
  const pre = document.createElement("pre");
  const codeElement = code(value);
  pre.style.whiteSpace = "pre-wrap";
  pre.style.overflowWrap = "anywhere";
  codeElement.style.whiteSpace = "pre-wrap";
  codeElement.style.overflowWrap = "anywhere";
  pre.append(codeElement);
  return pre;
};

const appendList = (parent, values) => {
  if (!values || values.length === 0) {
    return;
  }
  const list = document.createElement("ul");
  for (const value of values) {
    list.append(text("li", value));
  }
  parent.append(list);
};

const appendLinkedModuleList = (parent, moduleNames, reference) => {
  if (!moduleNames || moduleNames.length === 0) {
    return;
  }

  const list = document.createElement("ul");
  for (const moduleName of moduleNames) {
    const item = document.createElement("li");
    if (reference[moduleName]) {
      const link = text("a", moduleName);
      link.href = modulePage(moduleName);
      item.append(link);
    } else {
      item.append(text("span", moduleName));
    }
    list.append(item);
  }
  parent.append(list);
};

const appendKeyValueList = (parent, values) => {
  const entries = Object.entries(values || {});
  if (entries.length === 0) {
    return;
  }

  const list = document.createElement("dl");
  list.className = "param-list";
  for (const [name, value] of entries) {
    const term = document.createElement("dt");
    term.append(code(name));
    if (value && typeof value === "object" && value.Type) {
      term.append(" : ");
      term.append(text("span", value.Type));
    }
    if (value && typeof value === "object" && value.Default) {
      term.append(" = ");
      term.append(text("span", value.Default));
    }
    list.append(term);

    const description = document.createElement("dd");
    if (value && typeof value === "object") {
      description.textContent = value.Description || "";
    } else {
      description.textContent = String(value);
    }
    list.append(description);
  }
  parent.append(list);
};

const appendReturns = (parent, returns) => {
  if (!returns || Object.keys(returns).length === 0) {
    return;
  }

  const list = document.createElement("dl");
  list.className = "param-list";
  const term = document.createElement("dt");
  term.append(code("Returns"));
  if (returns.Type) {
    term.append(" : ");
    term.append(text("span", returns.Type));
  }
  list.append(term);
  list.append(text("dd", returns.Description || ""));
  parent.append(list);
};

const appendRaises = (parent, raises) => {
  if (!raises || raises.length === 0) {
    return;
  }
  parent.append(text("h4", "Raises"));
  appendList(parent, raises);
};

const appendSignatureBlock = (parent, signature) => {
  if (signature) {
    parent.append(codeBlock(signature));
  }
};

const appendCallableDetails = (parent, details) => {
  appendSignatureBlock(parent, details.Signature);
  if (details.Purpose) {
    parent.append(paragraph(details.Purpose));
  }

  if (details.Parameters && Object.keys(details.Parameters).length > 0) {
    parent.append(text("h4", "Parameters"));
    appendKeyValueList(parent, details.Parameters);
  }
  appendReturns(parent, details.Returns);
  appendRaises(parent, details.Raises);
  if (details.Example) {
    parent.append(text("h4", "Example"));
    parent.append(codeBlock(details.Example));
  }
};

const appendOverview = (parent, overview, reference) => {
  if (!overview) {
    return;
  }
  if (overview.Purpose) {
    parent.append(paragraph(overview.Purpose));
  }

  if (overview.Main_Use_Cases?.length) {
    parent.append(text("h4", "Main Use Cases"));
    appendList(parent, overview.Main_Use_Cases);
  }
  if (overview.Related_Modules?.length) {
    parent.append(text("h4", "Related Modules"));
    appendLinkedModuleList(parent, overview.Related_Modules, reference);
  }
};

const appendConstructor = (parent, constructor) => {
  if (!constructor || Object.keys(constructor).length === 0) {
    return;
  }
  parent.append(text("h4", "Constructor"));
  appendSignatureBlock(parent, constructor.Signature);
  if (constructor.Parameters && Object.keys(constructor.Parameters).length > 0) {
    appendKeyValueList(parent, constructor.Parameters);
  }
  appendRaises(parent, constructor.Raises);
};

const appendMethods = (parent, methods) => {
  const entries = Object.entries(methods || {});
  if (entries.length === 0) {
    return;
  }

  parent.append(text("h4", "Methods"));
  for (const [name, method] of entries) {
    const heading = document.createElement("h4");
    heading.append(code(name));
    parent.append(heading);
    appendCallableDetails(parent, method);
  }
};

const appendInternalMethods = (parent, methods) => {
  const entries = Object.entries(methods || {});
  if (entries.length === 0) {
    return;
  }
  parent.append(text("h4", "Internal Methods"));
  appendKeyValueList(parent, methods);
};

const appendClass = (parent, moduleName, className, classDetails) => {
  const section = document.createElement("section");
  section.id = `${slug(moduleName)}-${slug(className)}`;
  section.className = "api-entry";

  const heading = document.createElement("h3");
  heading.append(code(className));
  section.append(heading);

  if (classDetails.Purpose) {
    section.append(paragraph(classDetails.Purpose));
  }
  appendConstructor(section, classDetails.Constructor);

  if (classDetails.Attributes && Object.keys(classDetails.Attributes).length > 0) {
    section.append(text("h4", "Attributes"));
    appendKeyValueList(section, classDetails.Attributes);
  }
  appendMethods(section, classDetails.Methods);
  appendInternalMethods(section, classDetails.Internal_Methods);

  if (classDetails.Example) {
    section.append(text("h4", "Example"));
    section.append(codeBlock(classDetails.Example));
  }

  parent.append(section);
};

const appendFunction = (parent, moduleName, functionName, functionDetails) => {
  const section = document.createElement("section");
  section.id = `${slug(moduleName)}-${slug(functionName)}`;
  section.className = "api-entry";

  const heading = document.createElement("h3");
  heading.append(code(functionName));
  section.append(heading);
  appendCallableDetails(section, functionDetails);

  parent.append(section);
};

const appendInternalFunctions = (parent, moduleDetails) => {
  if (!moduleDetails.Internal_Functions) {
    return;
  }

  const section = document.createElement("section");
  section.className = "api-entry";
  section.id = "internal-functions";
  section.append(text("h3", "Internal Functions"));

  for (const [name, details] of Object.entries(moduleDetails.Internal_Functions)) {
    const heading = document.createElement("h4");
    heading.append(code(name));
    section.append(heading);

    if (details && typeof details === "object") {
      appendSignatureBlock(section, details.Signature);
      if (details.Purpose) {
        section.append(paragraph(details.Purpose));
      }
    } else {
      section.append(paragraph(String(details)));
    }
  }

  parent.append(section);
};

const appendTypeAliases = (parent, moduleDetails) => {
  if (!moduleDetails.Type_Aliases) {
    return;
  }

  const section = document.createElement("section");
  section.className = "api-entry";
  section.id = "type-aliases";
  section.append(text("h3", "Type Aliases"));
  appendKeyValueList(section, moduleDetails.Type_Aliases);
  parent.append(section);
};

const appendExceptions = (parent, moduleDetails) => {
  const exceptions = moduleDetails.Exceptions || {};
  if (Object.keys(exceptions).length === 0) {
    return;
  }

  const section = document.createElement("section");
  section.className = "api-entry";
  section.id = "exceptions";
  section.append(text("h3", "Exceptions"));
  appendKeyValueList(section, exceptions);
  parent.append(section);
};

const setLinkActive = (links, activeLink) => {
  for (const link of links) {
    const isActive = link === activeLink;
    link.classList.toggle("active", isActive);
    if (isActive) {
      link.setAttribute("aria-current", "true");
    } else {
      link.removeAttribute("aria-current");
    }
  }
};

const initScrollSpy = (linkSelector) => {
  const links = Array.from(document.querySelectorAll(linkSelector));
  const tracked = links
    .map((link) => ({
      link,
      section: document.querySelector(decodeURIComponent(link.hash)),
    }))
    .filter((item) => item.section);

  if (tracked.length === 0) {
    return undefined;
  }

  const updateActive = () => {
    const marker = window.scrollY + 120;
    const pageBottom =
      window.innerHeight + window.scrollY >=
      document.documentElement.scrollHeight - 24;
    let active = tracked[0];

    if (pageBottom) {
      active = tracked[tracked.length - 1];
    } else {
      for (const item of tracked) {
        if (item.section.offsetTop <= marker) {
          active = item;
        } else {
          break;
        }
      }
    }

    setLinkActive(links, active.link);
  };

  let ticking = false;
  const requestUpdate = () => {
    if (ticking) {
      return;
    }
    ticking = true;
    window.requestAnimationFrame(() => {
      updateActive();
      ticking = false;
    });
  };

  updateActive();
  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  return updateActive;
};

const appendModule = (moduleName, moduleDetails, reference) => {
  const moduleId = slug(moduleName);
  const section = document.createElement("section");
  section.id = moduleId;
  section.className = "api-module";

  const heading = document.createElement("h2");
  heading.append(code(moduleName));
  section.append(heading);
  appendOverview(section, moduleDetails.Overview, reference);

  const classes = Object.entries(moduleDetails.Classes || {});
  if (classes.length > 0) {
    section.append(text("h3", "Classes"));
    for (const [className, classDetails] of classes) {
      appendClass(section, moduleName, className, classDetails);
    }
  }

  const functions = Object.entries(moduleDetails.Functions || {});
  if (functions.length > 0) {
    section.append(text("h3", "Functions"));
    for (const [functionName, functionDetails] of functions) {
      appendFunction(section, moduleName, functionName, functionDetails);
    }
  }

  appendInternalFunctions(section, moduleDetails);
  appendTypeAliases(section, moduleDetails);
  appendExceptions(section, moduleDetails);
  modulesRoot.append(section);
};

const appendNavigation = (reference) => {
  const overviewLink = text("a", "Overview");
  overviewLink.href = "api.html";
  if (!currentModule) {
    overviewLink.classList.add("active");
    overviewLink.setAttribute("aria-current", "page");
  }
  moduleNav.append(overviewLink);

  for (const moduleName of Object.keys(reference)) {
    const moduleLink = text("a", moduleShortName(moduleName));
    moduleLink.href = modulePage(moduleName);
    if (moduleName === currentModule) {
      moduleLink.classList.add("active");
      moduleLink.setAttribute("aria-current", "page");
    }
    moduleNav.append(moduleLink);
  }
};

const appendOnPageNavigation = (moduleName, moduleDetails) => {
  const moduleId = slug(moduleName);
  const moduleLink = text("a", moduleShortName(moduleName));
  moduleLink.href = `#${moduleId}`;
  onPageNav.append(moduleLink);

  for (const className of Object.keys(moduleDetails.Classes || {})) {
    const classLink = text("a", className);
    classLink.href = `#${moduleId}-${slug(className)}`;
    onPageNav.append(classLink);
  }
  for (const functionName of Object.keys(moduleDetails.Functions || {})) {
    const functionLink = text("a", functionName);
    functionLink.href = `#${moduleId}-${slug(functionName)}`;
    onPageNav.append(functionLink);
  }
  if (moduleDetails.Internal_Functions) {
    const internalLink = text("a", "Internal Functions");
    internalLink.href = "#internal-functions";
    onPageNav.append(internalLink);
  }
  if (moduleDetails.Type_Aliases) {
    const typeLink = text("a", "Type Aliases");
    typeLink.href = "#type-aliases";
    onPageNav.append(typeLink);
  }
  if (Object.keys(moduleDetails.Exceptions || {}).length > 0) {
    const exceptionsLink = text("a", "Exceptions");
    exceptionsLink.href = "#exceptions";
    onPageNav.append(exceptionsLink);
  }
};

const renderIndex = (reference) => {
  const section = document.createElement("section");
  section.className = "api-module";
  section.append(text("h2", "Modules"));

  const list = document.createElement("div");
  list.className = "reference-list compact-reference api-index-list";

  for (const [moduleName, moduleDetails] of Object.entries(reference)) {
    const link = document.createElement("a");
    link.className = "reference-item reference-link";
    link.href = modulePage(moduleName);
    link.append(text("h3", moduleShortName(moduleName)));
    link.append(code(moduleName));
    if (moduleDetails.Overview?.Purpose) {
      link.append(paragraph(moduleDetails.Overview.Purpose));
    }
    list.append(link);
  }

  section.append(list);
  modulesRoot.append(section);
};

const setHeaderForIndex = () => {
  document.title = "API Reference - Evol Virtual Creature";
  if (pageTitle) {
    pageTitle.textContent = "Evol Virtual Creature API";
  }
  if (pageLede) {
    pageLede.innerHTML =
      'Module-oriented reference generated from <code>api-reference.json</code>. Select a module to open its reference.';
  }
};

const setHeaderForModule = (moduleName) => {
  document.title = `${moduleShortName(moduleName)} API - Evol Virtual Creature`;
  if (pageTitle) {
    pageTitle.textContent = moduleShortName(moduleName);
  }
  if (pageLede) {
    pageLede.replaceChildren(code(moduleName));
  }
};

const renderUnknownModule = (moduleName, reference) => {
  document.title = "Unknown API Module - Evol Virtual Creature";
  if (pageTitle) {
    pageTitle.textContent = "Unknown API module";
  }
  if (pageLede) {
    pageLede.replaceChildren(code(moduleName));
  }

  const section = document.createElement("section");
  section.className = "api-entry";
  section.append(text("h3", "Module not found"));
  section.append(
    paragraph(
      "The requested module is not present in api-reference.json. Select an available module from the API navigation."
    )
  );
  appendList(section, Object.keys(reference));
  modulesRoot.append(section);
};

const scrollToHashTarget = (updateOnPageNav) => {
  if (!location.hash) {
    return;
  }
  const target = document.querySelector(decodeURIComponent(location.hash));
  if (!target) {
    return;
  }
  const previousScrollBehavior = document.documentElement.style.scrollBehavior;
  document.documentElement.style.scrollBehavior = "auto";
  window.scrollTo({
    top: Math.max(target.offsetTop - 96, 0),
    behavior: "auto",
  });
  document.documentElement.style.scrollBehavior = previousScrollBehavior;
  updateOnPageNav?.();
};

const renderReference = (reference) => {
  appendNavigation(reference);

  if (!currentModule) {
    setHeaderForIndex();
    renderIndex(reference);
    return;
  }

  const moduleDetails = reference[currentModule];
  if (!moduleDetails) {
    renderUnknownModule(currentModule, reference);
    return;
  }

  setHeaderForModule(currentModule);
  appendOnPageNavigation(currentModule, moduleDetails);
  appendModule(currentModule, moduleDetails, reference);
  const updateOnPageNav = initScrollSpy("[data-api-on-page-nav] a[href^='#']");

  if (location.hash) {
    window.requestAnimationFrame(() => scrollToHashTarget(updateOnPageNav));
    window.setTimeout(() => scrollToHashTarget(updateOnPageNav), 150);
  }
};

const renderError = (error) => {
  const section = document.createElement("section");
  section.className = "api-entry";
  section.append(text("h3", "Could not load API reference"));
  section.append(
    paragraph(
      `The page could not load ${referencePath}. Serve the docs from a local web server or check that the JSON file is present.`
    )
  );
  section.append(codeBlock(String(error)));
  modulesRoot.append(section);
};

fetch(referencePath)
  .then((response) => {
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}`);
    }
    return response.json();
  })
  .then(renderReference)
  .catch(renderError);
