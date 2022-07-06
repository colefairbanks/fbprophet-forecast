<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://colefairbanks.com/forecasting">
    <img src="https://images.squarespace-cdn.com/content/v1/5a0b1aba90bcce4bd55450fb/ccdcf653-13ec-4d07-ab78-0920ff5dccc7/prophet_blue.jpg" alt="Logo">
  </a>

<h3 align="center">Facebook Prophet X Forecast</h3>

  <p align="center">
    <em>ADDITIVE LINEAR REGRESSION</em>
  </p>

  <p align="center">
    Advising Finance on Accurately Forecasting Revenue
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      </ul>
    </li>
    <li>
      <a href="#Prerequisites">Prerequisites</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

![Product Name Screen Shot](https://images.squarespace-cdn.com/content/v1/5a0b1aba90bcce4bd55450fb/5fdefbcf-4e6b-46b0-8c09-9c04dec8a98b/z_graph.png)

Company X's Finance Dept., while able to confidently predict X hardware (HW) sales two years out, has struggled to produce accurate revenue forecasts for the maintenance (MA) side of their business, which is similar to Apple Care but for legacy servers. This is a problem for Finance, as these forecasts are the basis for  long-term budgets as well as short-term business decisions.  As a Data Scientist familiar with financial data, we have been asked to forecast MA revenue for one of Company X's flagship brands, X.  The CFO is particularly interested in having reliable automated forecasts that extend beyond 6 months, with an ideal time horizon of 24 months and a consistent accuracy of 92% or above.  Fortunately, upon initial inspection, X’s historical 10-year monthly MA revenue seems to follow a predictable pattern that may allow us to deliver on the CFO's expectations.  The challenge, however, will be to factor in the relationship between HW, MA, as well as X’s unique 2-year product life cycle.

<p align="right">(<a href="#top">back to top</a>)</p>

## About Facebook Prophet

Prophet is an open source library published by Facebook that is based on decomposable (trend+seasonality+holidays) models. It provides Analysts the ability to create accurate time series forecasts and add regressors as well as the flexibility to adjust parameters like seasonality and holidays.  Prophet is especially suited for revenue and other business-oriented data that has clear seasonality and spans a few years on a monthly or weekly basis.

Regarding how Prophet was built, this alogrithm sees forecasting primarily as a curve fitting exercise using probabilistic techniques used in various well-known additive models like ARIMA. If you are interested in learning more about the mathematical components of Prophet, I suggest reading the original Prophet paper, [Forecasting at Scale](https://peerj.com/preprints/3190/), by Sean J Taylor, Benjamin Letham.

<p align="right">(<a href="#top">back to top</a>)</p>

## Prerequisites

### Installation

* fbprophet
  ```sh
  !pip install fbprophet
  ```

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Cole Fairbanks - ccole.fairbanks@gmail.com

Full Project Analysis: [https://github.com/colefairbanks/fbprophet-xforecast](https://github.com/colefairbanks/fbprophet-xforecast)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/colefairbanks/fbprophet-xforecast.svg?style=for-the-badge
[contributors-url]: https://github.com/colefairbanks/fbprophet-xforecast/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/colefairbanks/fbprophet-xforecast.svg?style=for-the-badge
[forks-url]: https://github.com/colefairbanks/fbprophet-xforecast/network/members
[stars-shield]: https://img.shields.io/github/stars/colefairbanks/fbprophet-xforecast.svg?style=for-the-badge
[stars-url]: https://github.com/colefairbanks/fbprophet-xforecast/stargazers
[issues-shield]: https://img.shields.io/github/issues/colefairbanks/fbprophet-xforecast.svg?style=for-the-badge
[issues-url]: https://github.com/colefairbanks/fbprophet-xforecast/issues
[license-shield]: https://img.shields.io/github/license/colefairbanks/fbprophet-xforecast.svg?style=for-the-badge
[license-url]: https://github.com/colefairbanks/fbprophet-xforecast/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/colefairbanks
[product-screenshot]: images/screenshot.png
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
